#include "cc.h"
#include <cmath>
#include <iomanip>
#include <numeric>
#include <fstream>

// =====================================================================
// Motion-tracking policy adaptor for whole_body_tracking/p73_walk.
//
// Policy ONNX: /home/user/ros2_ws/src/p73_cc/policy/policy_tracking.onnx
//   Inputs:
//     obs       [1, 80]   — single frame, no history stacking
//     time_step [1, 1]    — float frame index (cast to long internally)
//   Outputs:
//     actions         [1, 13]      — raw action, Isaac order
//     joint_pos       [1, 13]      — motion reference joint pos (aux)
//     joint_vel       [1, 13]      — motion reference joint vel (aux)
//     body_pos_w      [1, 8, 3]    — motion body positions, world frame (aux)
//     body_quat_w     [1, 8, 4]    — motion body quats, world frame (wxyz) (aux)
//     body_lin_vel_w  [1, 8, 3]    — (aux, unused)
//     body_ang_vel_w  [1, 8, 3]    — (aux, unused)
//
// The motion reference is baked into the ONNX via the `time_step` input —
// we warm up once at startup by calling the network over all frames and
// cache the slices we need (joint_pos/vel, anchor pos/quat at body index 0).
// =====================================================================

// =====================================================================
// Constructor
// =====================================================================
CustomController::CustomController(DataContainer &dc, RobotEigenData &rd)
    :   dc_(dc), rd_(rd),
        env(ORT_LOGGING_LEVEL_WARNING, "p73_cc"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr)
{
    weight_dir_ = "/home/user/ros2_ws/src/p73_cc/policy/policy_tracking.onnx";

    if (is_write_file_) {
        writeFile.open("/tmp/p73_cc_data.csv", ofstream::out);
        writeFile << fixed << setprecision(8);
    }

    loadOnnX();
    prefetchMotionCache();
    initVariable();
    startVelSubscriber();
}

// =====================================================================
// initVariable — defaults in both P73 and Isaac orders
// =====================================================================
void CustomController::initVariable()
{
    cout << "[p73_cc] Initializing variables" << endl;

    // MuJoCo / SHM / rd_.q_ order (13):
    //   L_HipRoll, L_HipPitch, L_HipYaw, L_Knee, L_AnklePitch, L_AnkleRoll,
    //   R_HipRoll, R_HipPitch, R_HipYaw, R_Knee, R_AnklePitch, R_AnkleRoll,
    //   WaistYaw
    q_default_p73_ << 0.0, 0.18, 0.0, 0.35, -0.17, 0.0,
                      0.0, -0.18, 0.0, -0.35, 0.17, 0.0,
                      0.0;

    // Isaac order (13), from ONNX metadata "default_joint_pos":
    //   [0, 0, 0, 0.18, -0.18, 0, 0, 0.35, -0.35, -0.17, 0.17, 0, 0]
    q_default_isaac_ << 0.0, 0.0, 0.0,
                        0.18, -0.18, 0.0, 0.0,
                        0.35, -0.35,
                        -0.17, 0.17,
                        0.0, 0.0;

    kp_p73_ << 1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
               1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
               576.0;

    kd_p73_ << 76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
               76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
               19.2;

    torque_bound_p73_ << 352.0, 220.0, 95.0, 220.0, 95.0, 95.0,
                         352.0, 220.0, 95.0, 220.0, 95.0, 95.0,
                         152.0;

    q_limit_lower_p73_ << -0.58, -1.57, -0.78, 0.0, -1.05, -0.42,
                          -0.58, -2.09, -0.78, -2.56, -0.7, -0.42;
    q_limit_upper_p73_ << 0.3, 2.09, 0.78, 2.56, 0.7, 0.42,
                          0.3, 1.57, 0.78, 0.0, 1.05, 0.42;

    rl_action_.setZero();
    last_action_raw_.setZero();
    torque_rl_.setZero();
    cached_anet_torque_.setZero();
    anet_hist_initialized_ = false;

    policy_frame_.assign(NUM_TRACKING_OBS, 0.0f);

    if (use_actuator_net_) {
        loadActuatorNets();
    }
}

// =====================================================================
// loadOnnX — binds obs + time_step inputs, actions + aux outputs
// =====================================================================
void CustomController::loadOnnX()
{
    cout << "[p73_cc] Loading tracking policy from " << weight_dir_ << endl;

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");
    session = Ort::Session(env, weight_dir_.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_number  = session.GetInputCount(); //2 -> obs, time_step
    output_number = session.GetOutputCount(); //7 -> actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_char.resize(input_number);
    output_names_char.resize(output_number);

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
        input_names[i] = name.get();
    }
    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = name.get();
    }

    cout << "[p73_cc] Inputs: ";
    copy(input_names.begin(), input_names.end(), ostream_iterator<string>(cout, " "));
    cout << endl;
    cout << "[p73_cc] Outputs: ";
    copy(output_names.begin(), output_names.end(), ostream_iterator<string>(cout, " "));
    cout << endl;

    for (size_t i = 0; i < input_names.size(); ++i) {
        input_names_char[i] = input_names[i].c_str();
        if (input_names[i] == "obs")        input_obs_idx_        = static_cast<int>(i);
        if (input_names[i] == "time_step")  input_time_step_idx_  = static_cast<int>(i);
    }
    for (size_t i = 0; i < output_names.size(); ++i) {
        output_names_char[i] = output_names[i].c_str();
        if (output_names[i] == "actions")     output_actions_idx_     = static_cast<int>(i);
        if (output_names[i] == "joint_pos")   output_joint_pos_idx_   = static_cast<int>(i);
        if (output_names[i] == "joint_vel")   output_joint_vel_idx_   = static_cast<int>(i);
        if (output_names[i] == "body_pos_w")  output_body_pos_idx_    = static_cast<int>(i);
        if (output_names[i] == "body_quat_w") output_body_quat_idx_   = static_cast<int>(i);
    }

    if (input_obs_idx_ < 0)
        throw std::runtime_error("[p73_cc] ONNX input 'obs' not found.");
    if (input_time_step_idx_ < 0)
        throw std::runtime_error("[p73_cc] ONNX input 'time_step' not found.");
    if (output_actions_idx_ < 0)
        throw std::runtime_error("[p73_cc] ONNX output 'actions' not found.");

    // Allocate input buffers to match declared shapes
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        cout << "[p73_cc]   input " << i << " '" << input_names[i] << "' shape: ";
        for (size_t k = 0; k < input_shape.size(); k++)
            cout << input_shape[k] << (k + 1 < input_shape.size() ? "x" : "");
        cout << endl;

        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0f);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }

    // Sanity: obs buffer must be 80 floats
    if (static_cast<int>(input_states_buffer[input_obs_idx_].size()) != NUM_TRACKING_OBS) {
        throw std::runtime_error("[p73_cc] obs input size mismatch (expected 80).");
    }

    cout << "[p73_cc] Policy loaded." << endl;
}

// =====================================================================
// prefetchMotionCache — call ONNX once per frame to cache motion data
//
// The actor's output does NOT depend on `time_step` (see exporter.py), so we
// sweep time_step over all frames with dummy obs and keep the aux outputs.
// =====================================================================
void CustomController::prefetchMotionCache()
{
    if (output_joint_pos_idx_ < 0 || output_joint_vel_idx_ < 0 ||
        output_body_pos_idx_  < 0 || output_body_quat_idx_ < 0)
    {
        throw std::runtime_error("[p73_cc] aux motion outputs not present in ONNX — cannot cache motion.");
    }

    // The ONNX clamps time_step at (total_frames - 1). Find total_frames by
    // probing: step time_step upward until the joint_pos output stops changing.
    // Probe up to a large ceiling; real motions are typically < 10000 frames.
    const int PROBE_LIMIT = 20000;
    std::vector<std::vector<float>> jp_rows;
    jp_rows.reserve(1024);

    auto run_with_ts = [&](int ts) -> std::vector<Ort::Value> {
        std::fill(input_states_buffer[input_obs_idx_].begin(),
                  input_states_buffer[input_obs_idx_].end(), 0.0f);
        input_states_buffer[input_time_step_idx_][0] = static_cast<float>(ts);
        return session.Run(
            Ort::RunOptions{nullptr},
            input_names_char.data(), input_tensors.data(), input_number,
            output_names_char.data(), output_number);
    };

    // Collect until joint_pos repeats the previous value (indicates clamp hit).
    std::vector<float> prev_jp(13, 0.0f);
    bool have_prev = false;
    int detected = 0;
    for (int ts = 0; ts < PROBE_LIMIT; ++ts) {
        auto out = run_with_ts(ts);
        const float *jp = out[output_joint_pos_idx_].GetTensorMutableData<float>();
        std::vector<float> row(jp, jp + 13);

        if (have_prev && row == prev_jp && ts > 0) {
            detected = ts;  // ts is one past the last valid frame
            break;
        }
        jp_rows.push_back(row);
        prev_jp = row;
        have_prev = true;
    }

    if (detected == 0) {
        throw std::runtime_error("[p73_cc] Could not detect motion length within probe limit.");
    }
    total_frames_ = detected;

    cout << "[p73_cc] Motion length detected: " << total_frames_ << " frames" << endl;

    motion_joint_pos_.resize(total_frames_, 13);
    motion_joint_vel_.resize(total_frames_, 13);
    motion_anchor_pos_w_.resize(total_frames_, 3);
    motion_anchor_quat_w_.resize(total_frames_, 4);

    // Re-sweep to also collect joint_vel + anchor pos/quat. (We already have
    // joint_pos in jp_rows but one more pass is cleaner and fast.)
    for (int t = 0; t < total_frames_; ++t) {
        auto out = run_with_ts(t);
        const float *jp = out[output_joint_pos_idx_].GetTensorMutableData<float>();
        const float *jv = out[output_joint_vel_idx_].GetTensorMutableData<float>();
        const float *bp = out[output_body_pos_idx_].GetTensorMutableData<float>();   // [1, 8, 3]
        const float *bq = out[output_body_quat_idx_].GetTensorMutableData<float>();  // [1, 8, 4]

        for (int j = 0; j < 13; ++j) {
            motion_joint_pos_(t, j) = jp[j];
            motion_joint_vel_(t, j) = jv[j];
        }
        // anchor = body index 0 (base_link)
        motion_anchor_pos_w_(t, 0) = bp[MOTION_ANCHOR_IDX * 3 + 0];
        motion_anchor_pos_w_(t, 1) = bp[MOTION_ANCHOR_IDX * 3 + 1];
        motion_anchor_pos_w_(t, 2) = bp[MOTION_ANCHOR_IDX * 3 + 2];
        // quat is (w, x, y, z) — IsaacLab convention
        motion_anchor_quat_w_(t, 0) = bq[MOTION_ANCHOR_IDX * 4 + 0];
        motion_anchor_quat_w_(t, 1) = bq[MOTION_ANCHOR_IDX * 4 + 1];
        motion_anchor_quat_w_(t, 2) = bq[MOTION_ANCHOR_IDX * 4 + 2];
        motion_anchor_quat_w_(t, 3) = bq[MOTION_ANCHOR_IDX * 4 + 3];
    }

    cout << "[p73_cc] Motion cache built: "
         << "joint_pos(" << total_frames_ << ",13) "
         << "joint_vel(" << total_frames_ << ",13) "
         << "anchor_pos_w(" << total_frames_ << ",3) "
         << "anchor_quat_w(" << total_frames_ << ",4)" << endl;
}

// =====================================================================
// processNoise — TOCABI sim2real pattern (unchanged)
// =====================================================================
void CustomController::processNoise()
{
    noise_time_cur_ = rd_.control_time_us_ / 1e6;

    if (is_on_robot_)
    {
        q_noise_ = rd_.q_;

        double dt = noise_time_cur_ - noise_time_pre_;
        if (dt > 0.0) {
            double sampling_freq = 1.0 / dt;
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(rd_.q_dot_, q_dot_lpf_, sampling_freq, lpf_cutoff_hz_);
        }
        q_vel_noise_ = rd_.q_dot_;
    }
    else
    {
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<> dis(-0.00001, 0.00001);

        for (int i = 0; i < MODEL_DOF; i++)
            q_noise_(i) = rd_.q_(i) + dis(gen);

        double dt = noise_time_cur_ - noise_time_pre_;
        if (dt > 0.0) {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / dt;
            double sampling_freq = 1.0 / dt;
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, sampling_freq, lpf_cutoff_hz_);
        }

        q_noise_pre_ = q_noise_;
    }

    noise_time_pre_ = noise_time_cur_;
}

// =====================================================================
// processObservation — build 80-D tracking obs
//
// Layout (all in Isaac joint order where applicable):
//   [0  : 13]  command: motion.joint_pos[t]
//   [13 : 26]  command: motion.joint_vel[t]
//   [26 : 29]  motion_anchor_pos_b (3)
//   [29 : 35]  motion_anchor_ori_b (6)  — first 2 cols of rotmat, row-major
//   [35 : 38]  base_lin_vel (body frame, 3)
//   [38 : 41]  base_ang_vel (body frame, 3)
//   [41 : 54]  joint_pos_rel (q - q_default, Isaac order, 13)
//   [54 : 67]  joint_vel_rel (Isaac order, 13)
//   [67 : 80]  last_action   (raw, Isaac order, 13)
// =====================================================================
void CustomController::processObservation()
{
    const int t = time_step_;

    // --- Robot base world pose ---
    Quaterniond robot_q;
    robot_q.x() = rd_.q_virtual_(3);
    robot_q.y() = rd_.q_virtual_(4);
    robot_q.z() = rd_.q_virtual_(5);
    robot_q.w() = rd_.q_virtual_(6);
    Vector3d robot_pos_w(rd_.q_virtual_(0), rd_.q_virtual_(1), rd_.q_virtual_(2));

    // Body-frame angular velocity (MuJoCo gyro / state estimator gives body frame directly).
    Vector3d ang_vel_b = rd_.q_dot_virtual_.segment<3>(3);

    // Body-frame linear velocity (world lin-vel rotated into body).
    Vector3d lin_vel_w = rd_.q_dot_virtual_.segment<3>(0);
    Vector3d lin_vel_b = quatRotateInverse(robot_q, lin_vel_w);

    // --- Motion anchor in world frame, with accumulated loop offset ---
    Vector3d motion_anchor_pos_w(
        static_cast<double>(motion_anchor_pos_w_(t, 0)) + motion_pos_offset_(0),
        static_cast<double>(motion_anchor_pos_w_(t, 1)) + motion_pos_offset_(1),
        static_cast<double>(motion_anchor_pos_w_(t, 2)) + motion_pos_offset_(2));
    Quaterniond motion_anchor_q_w(
        static_cast<double>(motion_anchor_quat_w_(t, 0)),  // w
        static_cast<double>(motion_anchor_quat_w_(t, 1)),  // x
        static_cast<double>(motion_anchor_quat_w_(t, 2)),  // y
        static_cast<double>(motion_anchor_quat_w_(t, 3))); // z

    // motion_anchor_pos_b = R_robot^{-1} * (motion_pos_w - robot_pos_w)
    Vector3d dp_w = motion_anchor_pos_w - robot_pos_w;
    Vector3d motion_anchor_pos_b = quatRotateInverse(robot_q, dp_w);

    // motion_anchor_ori_b = quat_inv(robot_q) * motion_q  → rotmat → first 2 cols, row-major
    Quaterniond motion_anchor_q_b = robot_q.inverse() * motion_anchor_q_w;
    Matrix3d R = motion_anchor_q_b.toRotationMatrix();

    // --- Joint state in Isaac order ---
    Matrix<double, 13, 1> q_isaac, qdot_isaac;
    for (int p = 0; p < 13; ++p) {
        int is = kP73ToIsaac_13[p];
        q_isaac(is)    = q_noise_(p);
        qdot_isaac(is) = q_vel_noise_(p);
    }
    Matrix<double, 13, 1> jp_rel = q_isaac - q_default_isaac_;

    // --- Build 80-D frame ---
    int idx = 0;

    // command = motion.joint_pos[t] ++ motion.joint_vel[t]  (Isaac order, native)
    for (int j = 0; j < 13; ++j) policy_frame_[idx++] = motion_joint_pos_(t, j);
    for (int j = 0; j < 13; ++j) policy_frame_[idx++] = motion_joint_vel_(t, j);

    // motion_anchor_pos_b
    policy_frame_[idx++] = static_cast<float>(motion_anchor_pos_b(0));
    policy_frame_[idx++] = static_cast<float>(motion_anchor_pos_b(1));
    policy_frame_[idx++] = static_cast<float>(motion_anchor_pos_b(2));

    // motion_anchor_ori_b: mat[..., :2] reshaped → PyTorch row-major flatten
    // → [R(0,0), R(0,1), R(1,0), R(1,1), R(2,0), R(2,1)]
    policy_frame_[idx++] = static_cast<float>(R(0, 0));
    policy_frame_[idx++] = static_cast<float>(R(0, 1));
    policy_frame_[idx++] = static_cast<float>(R(1, 0));
    policy_frame_[idx++] = static_cast<float>(R(1, 1));
    policy_frame_[idx++] = static_cast<float>(R(2, 0));
    policy_frame_[idx++] = static_cast<float>(R(2, 1));

    // base_lin_vel (body frame)
    policy_frame_[idx++] = static_cast<float>(lin_vel_b(0));
    policy_frame_[idx++] = static_cast<float>(lin_vel_b(1));
    policy_frame_[idx++] = static_cast<float>(lin_vel_b(2));

    // base_ang_vel (body frame)
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(0));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(1));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(2));

    // joint_pos_rel  (Isaac)
    for (int j = 0; j < 13; ++j) policy_frame_[idx++] = static_cast<float>(jp_rel(j));

    // joint_vel_rel  (Isaac) — no scaling, matches IsaacLab joint_vel_rel
    for (int j = 0; j < 13; ++j) policy_frame_[idx++] = static_cast<float>(qdot_isaac(j));

    // last_action (raw, Isaac order)
    for (int j = 0; j < 13; ++j) policy_frame_[idx++] = static_cast<float>(last_action_raw_(j));

    // Copy into ONNX input buffer
    std::memcpy(input_states_buffer[input_obs_idx_].data(),
                policy_frame_.data(), sizeof(float) * NUM_TRACKING_OBS);

    // Feed time_step (float index; ONNX casts to long and clamps internally)
    input_states_buffer[input_time_step_idx_][0] = static_cast<float>(t);
}

// =====================================================================
// feedforwardPolicy — run ONNX, extract 13-D actions
// =====================================================================
void CustomController::feedforwardPolicy()
{
    auto local_output = session.Run(
        Ort::RunOptions{nullptr},
        input_names_char.data(), input_tensors.data(), input_number,
        output_names_char.data(), output_number);

    if (output_actions_idx_ >= 0 &&
        static_cast<size_t>(output_actions_idx_) < local_output.size() &&
        local_output[output_actions_idx_].IsTensor())
    {
        const float *a = local_output[output_actions_idx_].GetTensorMutableData<float>();
        for (int i = 0; i < NUM_TRACKING_ACT; ++i)
            rl_action_(i) = static_cast<double>(a[i]);
    }

    // Diagnostic: dump first 30 policy calls to /tmp/p73_cc_diag.log
    static int diag_cnt = 0;
    static std::ofstream diag_file("/tmp/p73_cc_diag.log", std::ios::out);
    if (diag_cnt < 30 && diag_file.is_open()) {
        Eigen::IOFormat fmt(5, 0, " ", " ");
        diag_file << std::fixed << std::setprecision(5);
        diag_file << "=== step " << diag_cnt << " time_step=" << time_step_ << " ===\n";
        diag_file << "robot_pos_w: " << rd_.q_virtual_(0) << " " << rd_.q_virtual_(1) << " " << rd_.q_virtual_(2) << "\n";
        diag_file << "robot_quat_xyzw: " << rd_.q_virtual_(3) << " " << rd_.q_virtual_(4)
                  << " " << rd_.q_virtual_(5) << " " << rd_.q_virtual_(6) << "\n";
        diag_file << "motion_anchor_pos_w(+offset): "
                  << motion_anchor_pos_w_(time_step_, 0) + motion_pos_offset_(0) << " "
                  << motion_anchor_pos_w_(time_step_, 1) + motion_pos_offset_(1) << " "
                  << motion_anchor_pos_w_(time_step_, 2) + motion_pos_offset_(2) << "\n";
        diag_file << "motion_anchor_quat_wxyz: "
                  << motion_anchor_quat_w_(time_step_, 0) << " "
                  << motion_anchor_quat_w_(time_step_, 1) << " "
                  << motion_anchor_quat_w_(time_step_, 2) << " "
                  << motion_anchor_quat_w_(time_step_, 3) << "\n";
        diag_file << "obs[0..12]  cmd_jpos: ";
        for (int k = 0; k < 13; ++k) diag_file << policy_frame_[k] << " ";
        diag_file << "\n";
        diag_file << "obs[26..28] anchor_pos_b: "
                  << policy_frame_[26] << " " << policy_frame_[27] << " " << policy_frame_[28] << "\n";
        diag_file << "obs[29..34] anchor_ori_b: "
                  << policy_frame_[29] << " " << policy_frame_[30] << " "
                  << policy_frame_[31] << " " << policy_frame_[32] << " "
                  << policy_frame_[33] << " " << policy_frame_[34] << "\n";
        diag_file << "obs[35..37] lin_vel_b: "
                  << policy_frame_[35] << " " << policy_frame_[36] << " " << policy_frame_[37] << "\n";
        diag_file << "obs[38..40] ang_vel_b: "
                  << policy_frame_[38] << " " << policy_frame_[39] << " " << policy_frame_[40] << "\n";
        diag_file << "obs[41..53] jpos_rel: ";
        for (int k = 41; k <= 53; ++k) diag_file << policy_frame_[k] << " ";
        diag_file << "\n";
        diag_file << "obs[54..66] jvel_rel: ";
        for (int k = 54; k <= 66; ++k) diag_file << policy_frame_[k] << " ";
        diag_file << "\n";
        diag_file << "rl_action(isaac): " << rl_action_.transpose().format(fmt) << "\n\n";
        diag_file.flush();
        diag_cnt++;
    }

    // last_action in obs uses the RAW action (IsaacLab mdp.last_action).
    last_action_raw_ = rl_action_;
}

// =====================================================================
// computeFast — runs at 1 kHz, policy inference at 50 Hz
// =====================================================================
void CustomController::computeFast()
{
    float control_time_us = rd_.control_time_us_;

    static bool init = true;
    if (init) {
    // Robot's initial world pose (from keyframe / state estimator)
    Vector3d robot_pos0(rd_.q_virtual_(0), rd_.q_virtual_(1), rd_.q_virtual_(2));
    Quaterniond robot_q0;
    robot_q0.x() = rd_.q_virtual_(3);
    robot_q0.y() = rd_.q_virtual_(4);
    robot_q0.z() = rd_.q_virtual_(5);
    robot_q0.w() = rd_.q_virtual_(6);

    // Motion's frame-0 world pose
    Vector3d motion_pos0(static_cast<double>(motion_anchor_pos_w_(0, 0)),
                         static_cast<double>(motion_anchor_pos_w_(0, 1)),
                         static_cast<double>(motion_anchor_pos_w_(0, 2)));
    Quaterniond motion_q0(static_cast<double>(motion_anchor_quat_w_(0, 0)),   // w
                          static_cast<double>(motion_anchor_quat_w_(0, 1)),   // x
                          static_cast<double>(motion_anchor_quat_w_(0, 2)),   // y
                          static_cast<double>(motion_anchor_quat_w_(0, 3)));  // z

    // Yaw-only correction: extract yaw from motion_q0 and from robot_q0
    auto yawOf = [](const Quaterniond& q) {
        // ZYX: yaw = atan2(2(wz+xy), 1 - 2(y^2 + z^2))
        return std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                          1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    };
    double yaw_motion = yawOf(motion_q0);
    double yaw_robot  = yawOf(robot_q0);
    double dyaw = yaw_robot - yaw_motion;

    Quaterniond R_yaw(Eigen::AngleAxisd(dyaw, Vector3d::UnitZ()));

    for (int t = 0; t < total_frames_; ++t) {
        Vector3d p(static_cast<double>(motion_anchor_pos_w_(t, 0)),
                   static_cast<double>(motion_anchor_pos_w_(t, 1)),
                   static_cast<double>(motion_anchor_pos_w_(t, 2)));
        // pivot around motion frame-0 origin
        Vector3d p_rel = p - motion_pos0;
        Vector3d p_new = R_yaw * p_rel + robot_pos0;
        motion_anchor_pos_w_(t, 0) = static_cast<float>(p_new(0));
        motion_anchor_pos_w_(t, 1) = static_cast<float>(p_new(1));
        motion_anchor_pos_w_(t, 2) = static_cast<float>(p_new(2));

        Quaterniond q(static_cast<double>(motion_anchor_quat_w_(t, 0)),
                      static_cast<double>(motion_anchor_quat_w_(t, 1)),
                      static_cast<double>(motion_anchor_quat_w_(t, 2)),
                      static_cast<double>(motion_anchor_quat_w_(t, 3)));
        Quaterniond q_new = R_yaw * q;
        q_new.normalize();
        motion_anchor_quat_w_(t, 0) = static_cast<float>(q_new.w());
        motion_anchor_quat_w_(t, 1) = static_cast<float>(q_new.x());
        motion_anchor_quat_w_(t, 2) = static_cast<float>(q_new.y());
        motion_anchor_quat_w_(t, 3) = static_cast<float>(q_new.z());
    }
    motion_pos_offset_.setZero();  // offset already baked into the cache

    cout << "[p73_cc] Motion aligned: dyaw=" << dyaw
         << " rad, dp=" << (robot_pos0 - motion_pos0).transpose() << endl;
}

    // Per-tick noise/velocity update (1 kHz)
    processNoise();

    // Policy update at 50 Hz
    static int policy_step_count = 0;
    if ((control_time_us - time_inference_pre_) / 1.0e6 >= policy_dt_) {
        processObservation();
        feedforwardPolicy();
        time_inference_pre_ = control_time_us;
        policy_step_count++;

        // Advance motion frame and clamp at last frame (play once, then hold).
        int next = time_step_ + 1;
        if (next >= total_frames_) next = total_frames_ - 1;
        time_step_ = next;
    }

    // --- Action → target joint position (Isaac → P73) ---
    VectorQd target_pos = q_default_p73_;
    for (int is = 0; is < NUM_TRACKING_ACT; ++is) {
        int p = kIsaacToP73_13[is];
        double scaled_delta = rl_action_(is) * kActionScale[is];
        target_pos(p) = q_default_p73_(p) + scaled_delta;
    }
    // Clamp lower 12 joints to their P73 limits (waist has no limit record here)
    for (int p = 0; p < 12; ++p) {
        target_pos(p) = DyrosMath::minmax_cut(target_pos(p),
                                              q_limit_lower_p73_(p),
                                              q_limit_upper_p73_(p));
    }

    // --- Torque ---
    if (control_time_us < start_time_ + 0.1e6) {
        // 100 ms spline ramp from q_init to target — safety on mode entry.
        for (int i = 0; i < MODEL_DOF; i++) {
            rd_.q_desired(i) = DyrosMath::cubic(control_time_us,
                                                start_time_,
                                                start_time_ + 0.1e6,
                                                q_init_(i),
                                                target_pos(i),
                                                0.0, 0.0);
        }
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_rl_(i) = kp_p73_(i) * (rd_.q_desired(i) - q_noise_(i))
                          - kd_p73_(i) * q_vel_noise_(i);
        }
    } else if (use_actuator_net_) {
        rd_.q_desired = target_pos;
        computeActuatorNetTorques();
        torque_rl_ = cached_anet_torque_;
    } else {
        rd_.q_desired = target_pos;
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_rl_(i) = kp_p73_(i) * (rd_.q_desired(i) - q_noise_(i))
                          - kd_p73_(i) * q_vel_noise_(i);
        }
    }

    // --- Motor-space torque clamp via 4-bar (real robot) or local 4-bar (sim) ---
    if (is_on_robot_) {
        VectorQd torque_motor = WBC::JointTorqueToMotorTorque(rd_, torque_rl_);
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_motor(i) = DyrosMath::minmax_cut(torque_motor(i),
                                                    -torque_bound_p73_(i),
                                                    torque_bound_p73_(i));
        }
        rd_.torque_desired = torque_motor;
    } else {
        VectorQd q_motor_curr;
        sim_four_bar_.Joint2MotorDesiredPos(q_noise_, q_motor_curr);
        VectorQd joint_pos_dummy, joint_vel_dummy;
        VectorQd motor_vel_zero = VectorQd::Zero();
        sim_four_bar_.Motor2JointPosVel(q_motor_curr, joint_pos_dummy,
                                        motor_vel_zero, joint_vel_dummy);
        MatrixQQd J = sim_four_bar_.getFourBarJaco();

        VectorQd torque_motor = J.transpose() * torque_rl_;
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_motor(i) = DyrosMath::minmax_cut(torque_motor(i),
                                                    -torque_bound_p73_(i),
                                                    torque_bound_p73_(i));
        }
        rd_.torque_desired = J.transpose().inverse() * torque_motor;
    }

    // --- CSV log (adapted for 80D obs + 13D actions) ---
    static std::ofstream log_file;
    static bool log_opened = false;
    if (!log_opened) {
        std::string log_dir = std::string(getenv("HOME")) + "/ros2_ws/src/p73_cc/logs";
        if (is_on_robot_) log_dir = "/home/user/ros2_ws/src/p73_cc/logs";

        auto now = std::chrono::system_clock::now();
        auto t  = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&t, &tm_buf);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%y%m%d_%H%M%S", &tm_buf);
        std::string prefix = is_on_robot_ ? "realrobot" : "mujoco";
        std::string path = log_dir + "/" + prefix + "_tracking_" + ts + ".csv";
        log_file.open(path, std::ios::out);
        log_file << std::fixed << std::setprecision(8);

        log_file << "time,time_step";
        log_file << ",quat_x,quat_y,quat_z,quat_w";
        log_file << ",ang_vel_bx,ang_vel_by,ang_vel_bz";
        log_file << ",lin_vel_bx,lin_vel_by,lin_vel_bz";
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_raw_" << i;
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",qdot_" << i;
        for (int i = 0; i < NUM_TRACKING_OBS; i++) log_file << ",obs_" << i;
        for (int i = 0; i < NUM_TRACKING_ACT; i++) log_file << ",action_isaac_" << i;
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_joint_" << i;
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_motor_" << i;
        log_file << "\n";

        log_opened = true;
        cout << "[p73_cc] Logging to: " << path << endl;
    }

    if (log_file.is_open()) {
        VectorQd tau_joint = (control_time_us < start_time_ + 0.1e6) ? torque_spline_ : torque_rl_;
        Quaterniond qr; qr.x() = rd_.q_virtual_(3); qr.y() = rd_.q_virtual_(4);
                        qr.z() = rd_.q_virtual_(5); qr.w() = rd_.q_virtual_(6);
        Vector3d lin_vel_w = rd_.q_dot_virtual_.segment<3>(0);
        Vector3d lin_vel_b = quatRotateInverse(qr, lin_vel_w);
        Vector3d ang_vel_b = rd_.q_dot_virtual_.segment<3>(3);

        log_file << control_time_us / 1e6 << "," << time_step_;
        log_file << "," << qr.x() << "," << qr.y() << "," << qr.z() << "," << qr.w();
        log_file << "," << ang_vel_b(0) << "," << ang_vel_b(1) << "," << ang_vel_b(2);
        log_file << "," << lin_vel_b(0) << "," << lin_vel_b(1) << "," << lin_vel_b(2);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_(i);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << q_vel_noise_(i);
        for (int i = 0; i < NUM_TRACKING_OBS; i++) log_file << "," << policy_frame_[i];
        for (int i = 0; i < NUM_TRACKING_ACT; i++) log_file << "," << rl_action_(i);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << tau_joint(i);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.torque_desired(i);
        log_file << "\n";

        static int flush_cnt = 0;
        if (++flush_cnt % 100 == 0) log_file.flush();
    }

    // Debug console (~every 500 ms)
    static int dbg = 0;
    if (dbg++ % 500 == 0) {
        Eigen::IOFormat fmt(3, 0, " ", " ");
        cout << "[cc] t=" << control_time_us / 1e6
             << " frame=" << time_step_
             << " act: " << rl_action_.transpose().format(fmt) << endl;
    }
}

// =====================================================================
// loadActuatorNets — unchanged (disabled by default for tracking)
// =====================================================================
void CustomController::loadActuatorNets()
{
    std::string anet_path;
    if (is_on_robot_) {
        anet_path = "/home/user/ros2_ws/src/p73_cc/actuator_nets/actuator_nets.bin";
    } else {
        anet_path = std::string(getenv("HOME")) + "/ros2_ws/src/p73_cc/actuator_nets/actuator_nets.bin";
    }

    cout << "[p73_cc] Loading actuator nets from " << anet_path << endl;

    std::ifstream file(anet_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[p73_cc] Failed to open actuator net weights: " + anet_path);
    }

    auto read_matrix = [&](auto& mat) {
        constexpr int rows = std::remove_reference_t<decltype(mat)>::RowsAtCompileTime;
        constexpr int cols = std::remove_reference_t<decltype(mat)>::ColsAtCompileTime;
        float buf[rows * cols];
        file.read(reinterpret_cast<char*>(buf), sizeof(buf));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat(r, c) = static_cast<double>(buf[r * cols + c]);
    };

    auto read_vector = [&](auto& vec) {
        constexpr int size = std::remove_reference_t<decltype(vec)>::RowsAtCompileTime;
        float buf[size];
        file.read(reinterpret_cast<char*>(buf), sizeof(buf));
        for (int i = 0; i < size; i++)
            vec(i) = static_cast<double>(buf[i]);
    };

    for (int j = 0; j < 12; j++) {
        auto& w = anet_weights_[j];
        read_matrix(w.W0);
        read_vector(w.b0);
        read_matrix(w.W1);
        read_vector(w.b1);
        read_matrix(w.W2);
        read_vector(w.b2);
        read_matrix(w.W3);
        float b3_buf;
        file.read(reinterpret_cast<char*>(&b3_buf), sizeof(float));
        w.b3 = static_cast<double>(b3_buf);
    }

    if (file.fail()) {
        throw std::runtime_error("[p73_cc] Error reading actuator net weights (file truncated?)");
    }
}

double CustomController::anetForward(int j, const Eigen::Matrix<double, 6, 1>& input)
{
    const auto& w = anet_weights_[j];
    Eigen::Matrix<double, 32, 1> h = w.W0 * input + w.b0;
    h = h.array() / (1.0 + h.array().abs());
    h = w.W1 * h + w.b1;
    h = h.array() / (1.0 + h.array().abs());
    h = w.W2 * h + w.b2;
    h = h.array() / (1.0 + h.array().abs());
    return (w.W3 * h)(0) + w.b3;
}

void CustomController::computeActuatorNetTorques()
{
    // Legacy path for the 12-joint lower-body actuator net. Not used for
    // tracking (which expects 13-DOF actions incl. WaistYaw) — kept for
    // completeness; toggled off via use_actuator_net_ = false.
    float control_time_us = rd_.control_time_us_;
    static float anet_hist_time_pre = -1.0f;
    if (anet_hist_time_pre < 0.0f) anet_hist_time_pre = control_time_us - anet_dt_ * 1e6;

    bool do_hist_update = (control_time_us - anet_hist_time_pre) / 1.0e6 >= anet_dt_;
    if (do_hist_update) {
        for (int i = 0; i < 12; i++) {
            double pos_err = rd_.q_desired(i) - q_noise_(i);
            double vel = q_vel_noise_(i);
            if (!anet_hist_initialized_) {
                anet_pos_err_hist_[i] = {pos_err, pos_err};
                anet_vel_hist_[i] = {vel, vel};
            } else {
                anet_pos_err_hist_[i][1] = anet_pos_err_hist_[i][0];
                anet_pos_err_hist_[i][0] = pos_err;
                anet_vel_hist_[i][1] = anet_vel_hist_[i][0];
                anet_vel_hist_[i][0] = vel;
            }
        }
        anet_hist_initialized_ = true;
        anet_hist_time_pre = control_time_us;
    }

    for (int i = 0; i < 12; i++) {
        double pos_err_now = rd_.q_desired(i) - q_noise_(i);
        double vel_now = q_vel_noise_(i);
        Eigen::Matrix<double, 6, 1> anet_input;
        anet_input << pos_err_now, anet_pos_err_hist_[i][0], anet_pos_err_hist_[i][1],
                      vel_now, anet_vel_hist_[i][0], anet_vel_hist_[i][1];
        cached_anet_torque_(i) = anetForward(i, anet_input) * anet_output_scale_;
    }
    cached_anet_torque_(12) = kp_p73_(12) * (q_default_p73_(12) - q_noise_(12))
                            - kd_p73_(12) * q_vel_noise_(12);
}

// =====================================================================
void CustomController::computeSlow() {}

void CustomController::copyRobotData(RobotEigenData &rd_l)
{
    (void)rd_l;
}

Vector3d CustomController::quatRotateInverse(const Quaterniond &q, const Vector3d &v)
{
    Vector3d q_vec = q.vec();
    double q_w = q.w();
    Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Vector3d c = 2.0 * q_vec * q_vec.dot(v);
    return a - b + c;
}

// =====================================================================
// ROS2 Velocity Subscriber (kept for compatibility; unused by tracking policy)
// =====================================================================
void CustomController::velCmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(vel_mutex_);
    target_vel_x_   = msg->linear.x;
    target_vel_y_   = msg->linear.y;
    target_vel_yaw_ = msg->angular.z;
}

void CustomController::startVelSubscriber()
{
    vel_cbg_ = dc_.node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = vel_cbg_;

    vel_sub_ = dc_.node_->create_subscription<geometry_msgs::msg::Twist>(
        "/p73/cmd_vel", 10,
        std::bind(&CustomController::velCmdCallback, this, std::placeholders::_1),
        opts);

    vel_executor_.add_callback_group(vel_cbg_, dc_.node_->get_node_base_interface());

    vel_spin_running_ = true;
    vel_spin_thread_ = std::thread([this]() {
        while (vel_spin_running_ && rclcpp::ok()) {
            vel_executor_.spin_some(std::chrono::milliseconds(5));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    cout << "[p73_cc] (cmd_vel subscriber started — unused for tracking policy)" << endl;
}

void CustomController::stopVelSubscriber()
{
    vel_spin_running_ = false;
    if (vel_spin_thread_.joinable()) vel_spin_thread_.join();
    vel_sub_.reset();
}
