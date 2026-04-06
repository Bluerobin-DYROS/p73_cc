#include "cc.h"

// ANSI Color Codes for Debug Output
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"
#define COLOR_BOLD    "\033[1m"

// Optimized Debug Print Macros - only evaluate if debug enabled
#define DEBUG_HEADER(msg) if (debug_enabled_) { std::cout << COLOR_BOLD << COLOR_CYAN << "\n========== " << msg << " ==========" << COLOR_RESET << std::endl; }
#define DEBUG_SUCCESS(msg) if (debug_enabled_) { std::cout << COLOR_GREEN << "[✓] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_INFO(msg) if (debug_enabled_) { std::cout << COLOR_BLUE << "[INFO] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_WARN(msg) if (debug_enabled_) { std::cout << COLOR_YELLOW << "[WARN] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_ERROR(msg) if (debug_enabled_) { std::cout << COLOR_RED << "[ERROR] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_DATA(label, value) if (debug_enabled_) { std::cout << COLOR_MAGENTA << "[DATA] " << label << ": " << COLOR_WHITE << value << COLOR_RESET << std::endl; }

// const int kPolicyToRobot[CustomController::num_action] = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
const int kPolicyToRobot[CustomController::num_action] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
// const int kPolicyToRobot[CustomController::num_action] = {0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11};

CustomController::CustomController(DataContainer &dc, RobotEigenData &rd)
    : dc_(dc), rd_(rd)
{
    DEBUG_HEADER("INITIALIZING RL CONTROLLER");

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "br_humanoid");
    memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    DEBUG_SUCCESS("ONNX Runtime environment created");

    initVariable();

    // Load ONNX models
    std::cout << "Loading ONNX model..." << std::endl;
    std::cout << "Model paths: " << onnx_path << std::endl;
    loadOnnX(onnx_path);

    DEBUG_HEADER("LOADING ROS NODE");
    joy_sub_ = dc_.node_->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10,
        std::bind(&CustomController::joyCallback, this, std::placeholders::_1));
    DEBUG_SUCCESS("Joy subscriber created on topic: /joy");
}


CustomController::~CustomController()
{
    cout << "CustomController terminate" << creset << endl;
}

void CustomController::copyRobotData(RobotEigenData &rd_global_)
{
    memcpy(&rd_cc_, &rd_global_, sizeof(RobotEigenData));
}

void CustomController::computeFast()
{
    copyRobotData(rd_);

    if (dc_.task_cmd_.task_mode == 5)
    {
        if (!cc_mode_active_prev_)
        {
            cc_init_ = true;
            phase_started_ = true;
            cc_mode_active_prev_ = true;
        }

        if (cc_init_)
        {
            DEBUG_HEADER("INITIALIZING RL CONTROL LOOP");
            start_time_us = rd_cc_.control_time_us_;  

            q_noise_ = rd_cc_.q_;
            q_vel_noise_ = rd_cc_.q_dot_;
            q_init_  = rd_cc_.q_;
            
            time_cur_s = rd_cc_.control_time_;
            time_pre_s = rd_cc_.control_time_;
            time_init_s = rd_cc_.control_time_;

            time_inference_pre_us = start_time_us - del_t * 1.0e6;
            cc_init_ = false;

            DEBUG_DATA("Task mode", dc_.task_cmd_.task_mode);
            DEBUG_DATA("Current time", time_cur_s);
            DEBUG_DATA("Inference frequency", hz_);

            torque_init_ = rd_cc_.torque_desired;
            DEBUG_SUCCESS("Initial torque saved");

            DEBUG_INFO("Processing initial observation...");
            processNoise();
            processObservation();
            DEBUG_SUCCESS("RL Controller initialized successfully!");
            return; // Skip rest of computation on init frame
        }

        processNoise();
        if ((rd_cc_.control_time_us_ - time_inference_pre_us) >= (del_t * 1.0e6))
        {
            processObservation();
            feedforwardPolicy();
        }

        torque_rl_.setZero();
        
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_rl_(i) = rd_.Kp_j[i] * (q_init_(i) - rd_.q_(i)) + rd_.Kd_j[i] * (0.0 - rd_.q_dot_(i));
        }

        for (int i = 0; i < num_action; i++) {
            double target = q_init_(i) + rl_action_(i);
            target = DyrosMath::minmax_cut(target, q_min(i), q_max(i));

            torque_rl_(i) = rd_.Kp_j[i] * (target - rd_.q_(i)) + rd_.Kd_j[i] * (0.0 - rd_.q_dot_(i));
        }

        // const double torque_blend_time_s = 1.0;
        // for (int i = 0; i < num_action; i++)
        // {
        //     torque_rl_(i) = DyrosMath::cubic(
        //         rd_cc_.control_time_,
        //         time_init_s,
        //         time_init_s + torque_blend_time_s,
        //         torque_init_(i),
        //         torque_rl_(i),
        //         0.0,
        //         0.0
        //     );
        // }

        rd_.torque_desired = torque_rl_;
        
        if(!dc_.simMode){
            rd_.torque_desired = WBC::JointTorqueToMotorTorque(rd_, torque_rl_);
        }
    }
    else{
        if (cc_mode_active_prev_)
        {
            cc_mode_active_prev_ = false;
            cc_init_ = true;
            phase_started_ = true;
        }
        std::cout << "Unsupported task mode for RL control: " << dc_.task_cmd_.task_mode << std::endl;
    }

}

//--- RL Functions
void CustomController::processNoise()
{
    time_cur_s = rd_cc_.control_time_;  // Keep in seconds for time calculations
    const double time_delta = time_cur_s - time_pre_s;
    const bool valid_time_delta = (time_delta > 0.0 && time_delta < 1.0);  // Sanity check: time_delta should be < 1 second

    if (is_on_robot_)
    {
        q_noise_.noalias() = rd_cc_.q_;
        q_vel_noise_.noalias() = rd_cc_.q_dot_;
    }
    else
    {
        // In simulation: add noise to joint positions
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-0.00001, 0.00001);

        q_noise_.noalias() = rd_cc_.q_;
        q_vel_noise_.noalias() = rd_cc_.q_dot_;

        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) += dis(gen);
        }
    }

    time_pre_s = time_cur_s;
}

void CustomController::processObservation()
{
    int data_idx = 0;
    double phase_sin = 0.0;
    double phase_cos = 1.0;
    
    // std::cout << "===== OBSERVATION START =====" << std::endl;

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(6);
    // std::cout << "q: " << q.coeffs().transpose() << std::endl;

    Eigen::Vector3d euler_angle_ = DyrosMath::rot2Euler(q.toRotationMatrix());

    // (1) Root Height : Dim 1
    state_cur_[data_idx++] = rd_cc_.link_[Pelvis].xpos(2);
    // std::cout << "Root Height: " << rd_cc_.link_[Pelvis].xpos(2) << std::endl;

    // (2) Base Linear Velocity (body frame): Dim 3
    Eigen::Vector3d base_lin_vel_bf; 
    base_lin_vel_bf = DyrosMath::quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0, 3));
    
    state_cur_[data_idx++] = base_lin_vel_bf(0);
    state_cur_[data_idx++] = base_lin_vel_bf(1);
    state_cur_[data_idx++] = base_lin_vel_bf(2);
    // std::cout << "Base Linear Velocity (body frame): " << base_lin_vel_bf.transpose() << std::endl;

    // (3) Base Heading (yaw) : Dim 1
    // state_cur_[data_idx++] = euler_angle_(2);
    // std::cout << "Base Heading (yaw): " << euler_angle_(2) << std::endl;

    // (4) Base Angular Velocity (body frame): Dim 3
    Eigen::Vector3d base_ang_vel_bf;
    base_ang_vel_bf = DyrosMath::quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3, 3));

    state_cur_[data_idx++] = base_ang_vel_bf(0);
    state_cur_[data_idx++] = base_ang_vel_bf(1);
    state_cur_[data_idx++] = base_ang_vel_bf(2);
    // std::cout << "Base Angular Velocity (body frame): " << base_ang_vel_bf.transpose() << std::endl;


    // (5) Projected Gravity (body frame): Dim 3
    Eigen::Vector3d gravity_bf = DyrosMath::quatRotateInverse(q, Eigen::Vector3d(0.0, 0.0, -1.0));
    state_cur_[data_idx++] = gravity_bf(0);
    state_cur_[data_idx++] = gravity_bf(1);
    state_cur_[data_idx++] = gravity_bf(2);
    // std::cout << "Projected Gravity (body frame): " << gravity_bf.transpose() << std::endl;

    // (6) velocity_commands : Dim 3
    commands_.setZero();
    state_cur_[data_idx++] = commands_(0);
    state_cur_[data_idx++] = commands_(1);
    state_cur_[data_idx++] = commands_(2);
    // std::cout << "Velocity Commands: " << commands_.transpose() << std::endl;

    // (7) Phase Information (sin, cos): Dim 2
    if (phase_period_s_ > 0.0)
    {
        const double two_pi = 2 * M_PI;

        if (phase_started_)
        {
            phase_time_s_init_ = rd_cc_.control_time_;

            phase_started_ = false;
        }

        phase_time_s_ = rd_cc_.control_time_ - phase_time_s_init_;

        // double t = phase_started_ ? (phase_time_s_ + phase_offset_s_) : phase_offset_s_;
        double t = phase_time_s_ + phase_offset_s_;
        double phase = two_pi * (t / phase_period_s_);
        phase_sin = std::sin(phase);
        phase_cos = std::cos(phase);
        state_cur_[data_idx++] = phase_sin;
        state_cur_[data_idx++] = phase_cos;
    }
    else
    {
        phase_sin = 0.0;
        phase_cos = 1.0;
        state_cur_[data_idx++] = phase_sin;
        state_cur_[data_idx++] = phase_cos;
    }
    // std::cout << "Phase (sin, cos): " << phase_sin << ", " << phase_cos << std::endl;

    // joint_pos (legs) : Dim 12
    for (int i = 0; i < num_action; i++)
    {
        state_cur_[data_idx++] = q_noise_(kPolicyToRobot[i]);
    }
    // std::cout << "Joint Positions (legs): " << q_noise_.segment(0, 12).transpose() << std::endl;

    // joint_vel (legs) : Dim 12
    for (int i = 0; i < num_action; i++)
    {
        state_cur_[data_idx++] = q_vel_noise_(kPolicyToRobot[i]);
    }
    // std::cout << "Joint Velocities (legs): " << q_vel_noise_.segment(0, 12).transpose() << std::endl;
    
    // last_leg_action : Dim 12
    for (int i = 0; i < num_action; i++)
    {
        // state_cur_[data_idx++] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        state_cur_[data_idx++] = rl_action_(i);
    }
    // std::cout << "Last Leg Actions: " << rl_action_.transpose() << std::endl;
    
    // Centroidal Angular Momentum (body frame) : Dim 3
    // Eigen::Vector3d centroidal_angular_momentum_bf = DyrosMath::quatRotateInverse(q, rd_cc_.centroidal_angular_momentum_);
    // state_cur_[data_idx++] = centroidal_angular_momentum_bf(0);
    // state_cur_[data_idx++] = centroidal_angular_momentum_bf(1);
    // state_cur_[data_idx++] = centroidal_angular_momentum_bf(2);
    // state_cur_[data_idx++] = rd_cc_.centroidal_angular_momentum_(2); 
    // std::cout << "Centroidal Angular Momentum (body frame): " << centroidal_angular_momentum_bf(0) << ", " << centroidal_angular_momentum_bf(1) << ", " << rd_cc_.centroidal_angular_momentum_(2) << std::endl;

    if (data_idx != num_cur_state)
    {
        DEBUG_ERROR("Observation data index mismatch: expected " + std::to_string(num_cur_state) + ", got " + std::to_string(data_idx));
        assert(data_idx == num_cur_state);
    }

    // ===== VALIDATION: Check for NaN/Inf in observations =====
    bool obs_has_invalid = false;
    int first_invalid_idx = -1;
    for (int i = 0; i < num_cur_state; i++) {
        if (!std::isfinite(state_cur_[i])) {
            if (!obs_has_invalid) {
                first_invalid_idx = i;
            }
            obs_has_invalid = true;
        }
    }

    if (obs_has_invalid) {
        RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
            "NaN/Inf detected in observation at index %d! Resetting to safe values.", first_invalid_idx);
        // Replace invalid values with zeros as a safety measure
        for (int i = 0; i < num_cur_state; i++) {
            if (!std::isfinite(state_cur_[i])) {
                state_cur_[i] = 0.0f;
            }
        }
    }

    // ===== VALIDATION: Check that observations are not all zeros =====
    bool all_zeros = true;
    for (int i = 0; i < num_cur_state; i++) {
        if (std::abs(state_cur_[i]) > 1e-9) {
            all_zeros = false;
            break;
        }
    }

    if (all_zeros) {  // Skip first call during initialization
        RCLCPP_WARN_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 5000,
            "WARNING: All observations are zero! This indicates a potential bug in observation construction.");
    }

    //--- Save Input Buffer
    if (input_states_buffer.empty())
    {
        RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
            "input_states_buffer is empty. ONNX inputs may not be initialized.");
        return;
    }

    if (input_obs_idx_ < 0 || input_obs_idx_ >= static_cast<int>(input_states_buffer.size()))
    {
        RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
            "Invalid obs input index: %d (buffer size: %zu)", input_obs_idx_, input_states_buffer.size());
        return;
    }

    size_t obs_size = input_states_buffer[input_obs_idx_].size();
    if (obs_size == static_cast<size_t>(num_cur_state))
    {
        std::copy(state_cur_.begin(), state_cur_.end(), input_states_buffer[input_obs_idx_].begin());
        return;
    }

    RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
        "Observation size mismatch: obs tensor size=%zu, expected=%d. Observation is NOT copied.",
        obs_size, num_cur_state);
}

void CustomController::feedforwardPolicy()
{
    static int action_log_counter = 0;

    auto* actor_session = getSession("actor");
    if (!actor_session)
    {
        RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
            "Actor session is null. Skipping inference.");
        return;
    }

    output_tensors = actor_session->Run(Ort::RunOptions{nullptr},
                                        input_names_char.data(),
                                        input_tensors.data(),
                                        input_number,
                                        output_names_char.data(),
                                        output_number);

    if (output_tensors.empty())
    {
        DEBUG_ERROR("ONNX output_tensors is empty.");
        return;
    }

    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    if (output_action_idx_ < 0 || output_action_idx_ >= static_cast<int>(output_tensors.size()))
    {
        RCLCPP_ERROR_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
            "Invalid action output index: %d (output tensor count: %zu)", output_action_idx_, output_tensors.size());
        return;
    }

    // output tensor to rl_action_
    const float *action_data = output_tensors[output_action_idx_].GetTensorMutableData<float>();
    Ort::TypeInfo action_info = output_tensors[output_action_idx_].GetTypeInfo();
    auto action_tensor_info = action_info.GetTensorTypeAndShapeInfo();
    size_t action_count = action_tensor_info.GetElementCount();
    if (action_count < num_action)
    {
        DEBUG_ERROR("Action Dim Mismatch");
        std::cout << "ONNX action output has " << action_count << " elements; expected at least " << num_action << "." << std::endl;
        return;
    }
    for (size_t i = 0; i < num_action; i++) {
            // rl_action_(i) = DyrosMath::minmax_cut(action_data[i], -1.0, 1.0);
            rl_action_(i) = action_data[i];
    }

    time_inference_pre_us = rd_cc_.control_time_us_;
}

//--- INITIALIZATION
void CustomController::initVariable()
{
    DEBUG_HEADER("INITIALIZING VARIABLES");

    // Initialize action vector
    rl_action_.setZero(num_action);
    DEBUG_SUCCESS("Action vector initialized: " + std::to_string(num_action) + " dimensions");

    // Initialize state vectors for RL observation (using assign for better performance)
    state_cur_.assign(num_cur_state, 0.0f);
    normalized_state_cur_.assign(num_cur_state, 0.0f);
    h_cur_.assign(num_cur_h, 0.0f);
    DEBUG_SUCCESS("State vectors initialized");
    DEBUG_DATA("  - state_cur", num_cur_state);
    DEBUG_DATA("  - h_cur", num_cur_h);

    // Initialize robot state variables
    q_init_.setZero();
    q_noise_.setZero();
    q_noise_pre_.setZero();
    q_vel_noise_.setZero();

    q_min.setZero();
    q_max.setZero();
    double soft_joint_pos_limit_factor = 0.9;
    for(int i = 0; i < MODEL_DOF; i++)
    {
        double mean  = 0.5 * (rd_.q_min(i) + rd_.q_max(i));
        double range = 0.5 * (rd_.q_max(i) - rd_.q_min(i)) * soft_joint_pos_limit_factor;
        q_min(i) = mean - range;
        q_max(i) = mean + range;
    }
    std::cout << "Joint limits (q_min): " << q_min.transpose() << std::endl;
    std::cout << "Joint limits (q_max): " << q_max.transpose() << std::endl;

    torque_init_.setZero();
    torque_spline_.setZero();
    torque_rl_.setZero();
    DEBUG_SUCCESS("Robot state variables initialized");

    // Initialize velocity tracking
    base_lin_vel_.setZero();
    base_ang_vel_.setZero();
    DEBUG_SUCCESS("Velocity tracking initialized");

    // Initialize walking/stepping variables
    // Match IsaacLab config semantics: step_period is half-cycle, full phase cycle is 2*step_period.
    // With training config (step_period=18, dt=0.005, decimation=4): full cycle = 36*0.02 = 0.72 s.
    // step_period_s_ = 0.36;
    step_period_s_ = 0.7;
    phase_period_s_ = 2.0 * step_period_s_;
    step_ticks_ = 0.0;
    phase_indicator_ = 0;
    DEBUG_DATA("Step period", step_period_s_);
    DEBUG_DATA("Phase period", phase_period_s_);

    // Initialize command and control variables
    commands_.setZero();
    DEBUG_SUCCESS("Command variables initialized");

    // Initialize joystick velocity scaling
    vel_scale_x_ = 0.3;
    vel_scale_y_ = 0.3;
    vel_scale_yaw_ = 0.3;
    DEBUG_DATA("Velocity scales (x, y, yaw)", std::to_string(vel_scale_x_) + ", " + std::to_string(vel_scale_y_) + ", " + std::to_string(vel_scale_yaw_));

    // Initialize time tracking
    del_t = 1.0 / hz_;  // Time step based on control frequency
    DEBUG_DATA("Time step (del_t)", del_t);
}

void CustomController::loadOnnX(const std::string& model_path)
{
    DEBUG_HEADER("LOADING ONNX MODELS");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");
    session_options.SetIntraOpNumThreads(2);  // Optimize thread usage
    session_options.SetInterOpNumThreads(1);
    DEBUG_SUCCESS("ONNX session options configured");

    const auto& logger = dc_.node_->get_logger();

    try {
        std::filesystem::path resolved_model_path(model_path);
        if (resolved_model_path.is_relative()) {
            const std::string package_share_dir = ament_index_cpp::get_package_share_directory("p73_cc");
            resolved_model_path = std::filesystem::path(package_share_dir) / "onnx" / resolved_model_path;
        }

        if (!std::filesystem::exists(resolved_model_path)) {
            RCLCPP_ERROR(logger, "ONNX model file not found: %s", resolved_model_path.string().c_str());
            return;
        }

        DEBUG_INFO("Loading actor model from: " + resolved_model_path.string());
        sessions_["actor"] = std::make_unique<Ort::Session>(*env, resolved_model_path.string().c_str(), session_options);
        DEBUG_SUCCESS("Actor model loaded successfully");
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(logger, "Failed to load actor model: %s", e.what());
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    //--- Call Actor Netrowk
    DEBUG_HEADER("CONFIGURING ACTOR NETWORK");
    auto* actor_session = getSession("actor");
    if(!actor_session) {
        DEBUG_ERROR("Actor session not found");
        RCLCPP_ERROR(logger, "Actor session not found");
        return;
    }

    DEBUG_SUCCESS("Actor session found");

    input_obs_idx_ = -1;
    output_action_idx_ = -1;
    input_states_buffer.clear();
    input_tensors.clear();

    input_number = sessions_["actor"]->GetInputCount();
    output_number = sessions_["actor"]->GetOutputCount();
    DEBUG_DATA("Actor inputs", input_number);
    DEBUG_DATA("Actor outputs", output_number);

    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_char.resize(input_number);
    output_names_char.resize(output_number);

    // Get actor input names
    for (size_t i = 0; i < input_number; ++i) {
        Ort::AllocatedStringPtr input_name = actor_session->GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
    }

    // Get actor output names
    for (size_t i = 0; i < output_number; ++i) {
        Ort::AllocatedStringPtr output_name = actor_session->GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
    }

    // Print actor input/output names with color
    std::cout << COLOR_BOLD << COLOR_GREEN << "Actor Input names: " << COLOR_WHITE;
    std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << COLOR_RESET << std::endl;

    std::cout << COLOR_BOLD << COLOR_GREEN << "Actor Output names: " << COLOR_WHITE;
    std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << COLOR_RESET << std::endl;

    // Identify specific input/output indices
    for (size_t i = 0; i < input_names.size(); ++i) {
        input_names_char[i] = input_names[i].c_str();
        if (input_names[i] == "obs") {
            input_obs_idx_ = i;
            DEBUG_DATA("  Input 'obs' at index", i);
        } 
    }

    for (size_t i = 0; i < output_names.size(); ++i) {
        output_names_char[i] = output_names[i].c_str();
        if (output_names[i] == "action") {
            output_action_idx_ = i;
            DEBUG_DATA("  Output 'action' at index", i);
        } else if (output_names[i] == "23") {
            output_action_idx_ = i;
            DEBUG_DATA("  Output '23' at index", i);
        }
    }

    if (input_obs_idx_ < 0)
    {
        RCLCPP_ERROR(logger, "Could not find required input tensor named 'obs'.");
        return;
    }

    if (output_action_idx_ < 0)
    {
        RCLCPP_ERROR(logger, "Could not find required output tensor named 'action' (or fallback '23').");
        return;
    }

    // Initialize actor input tensors
    DEBUG_INFO("Initializing actor input tensors...");
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = actor_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        std::cout << COLOR_MAGENTA << "  Actor Input " << i << " (" << input_names[i] << ") shape: " << COLOR_WHITE;
        for (const auto& dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << COLOR_RESET << std::endl;

        size_t element_count = computeElementCount(input_shape);
        if (element_count == 0) {
            RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in actor input %zu", i);
            return;
        }
        std::vector<float> input_tensor_values(element_count, 0.0f);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            *memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }

    // Print actor output tensor shapes
    DEBUG_INFO("Checking actor output tensor shapes...");
    for (size_t i = 0; i < output_number; ++i) {
        Ort::TypeInfo type_info = actor_session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = tensor_info.GetShape();

        std::cout << COLOR_MAGENTA << "  Actor Output " << i << " (" << output_names[i] << ") shape: " << COLOR_WHITE;
        for (const auto& dim : output_shape) {
            std::cout << dim << " ";
        }
        std::cout << COLOR_RESET << std::endl;
    }

    DEBUG_SUCCESS("Actor input tensors initialized");
}

Ort::Session* CustomController::getSession(const std::string& name)
{
    const auto it = sessions_.find(name);
    return (it != sessions_.end()) ? it->second.get() : nullptr;
}

size_t CustomController::computeElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (const auto& dim : shape) {
        if (dim <= 0) {
            // Dynamic dimension (like batch size -1), use 1 as default
            DEBUG_WARN("Dynamic dimension detected in shape, using 1 as placeholder for batch dimension");
            count *= 1;  // Assume batch size of 1 for dynamic dimensions
        } else {
            count *= static_cast<size_t>(dim);
        }
    }
    return count;
}

//--- ROS2 Callbacks
void CustomController::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Check if we have enough axes and buttons
    if (msg->axes.size() < 2 || msg->buttons.size() < 8) {
        RCLCPP_WARN_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
            "Joy message does not have enough axes or buttons");
        return;
    }
    
    // Update velocity commands from joystick axes
    commands_(0) = DyrosMath::minmax_cut(vel_scale_x_ * msg->axes[1], -1.0, 1.0);
    commands_(1) = DyrosMath::minmax_cut(vel_scale_y_ * msg->axes[0], -1.0, 1.0);
    
    // Button 6: Turn right
    if (msg->buttons[6] == 1) {
        commands_(2) = vel_scale_yaw_;
    }
    // Button 7: Turn left
    else if (msg->buttons[7] == 1) {
        commands_(2) = -vel_scale_yaw_;
    }
    // No turn buttons pressed
    else {
        commands_(2) = 0.0;
    }
}

