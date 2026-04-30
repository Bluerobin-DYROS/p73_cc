#ifndef p73_cc_H
#define p73_cc_H

#include "p73_lib/robot_data.h"
#include "p73_lib/4bar_jac_func.h"
#include "wholebody_functions.h"
#include "onnxruntime_cxx_api.h"
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>

using namespace Eigen;
using namespace std;

class CustomController
{
public:
    CustomController(DataContainer &dc, RobotEigenData &rd);

    void computeSlow();
    void computeFast();
    void copyRobotData(RobotEigenData &rd_l);

    DataContainer &dc_;
    RobotEigenData &rd_;
    RobotEigenData rd_cc_;

    ofstream writeFile;
    bool is_write_file_ = false;

    //////////////////////// Functions ////////////////////////
    void initVariable();
    void loadOnnX();
    void prefetchMotionCache();
    void processNoise();
    void processObservation();
    void feedforwardPolicy();
    Vector3d quatRotateInverse(const Quaterniond &q, const Vector3d &v);

    //////////////////////// Joint Order Permutation (13 DOF) ////////////////////////
    // Isaac order (tracking policy ONNX):
    //   [0]L_HipRoll, [1]R_HipRoll, [2]WaistYaw,
    //   [3]L_HipPitch, [4]R_HipPitch, [5]L_HipYaw, [6]R_HipYaw,
    //   [7]L_Knee, [8]R_Knee,
    //   [9]L_AnklePitch, [10]R_AnklePitch,
    //   [11]L_AnkleRoll, [12]R_AnkleRoll
    //
    // MuJoCo / SHM / rd_.q_ order (this is what the p73 robot data actually uses;
    // see the NOTE at the top of cc.cpp):
    //   [0]L_HipRoll, [1]L_HipPitch, [2]L_HipYaw, [3]L_Knee, [4]L_AnklePitch, [5]L_AnkleRoll,
    //   [6]R_HipRoll, [7]R_HipPitch, [8]R_HipYaw, [9]R_Knee, [10]R_AnklePitch, [11]R_AnkleRoll,
    //   [12]WaistYaw
    //
    // kIsaacToP73_13[isaac_idx] = mujoco_idx  (for writing Isaac-order values into rd_.q_-order arrays)
    // kP73ToIsaac_13[mujoco_idx] = isaac_idx  (for reading rd_.q_-order values into Isaac-order obs)
    static constexpr std::array<int, 13> kIsaacToP73_13 = {
        0,  6, 12,      // L_HipRoll, R_HipRoll, WaistYaw
        1,  7,  2,  8,  // L_HipPitch, R_HipPitch, L_HipYaw, R_HipYaw
        3,  9,          // L_Knee, R_Knee
        4, 10,          // L_AnklePitch, R_AnklePitch
        5, 11           // L_AnkleRoll, R_AnkleRoll
    };
    static constexpr std::array<int, 13> kP73ToIsaac_13 = {
        0,  3,  5,  7,  9, 11,  // L leg (Roll, Pitch, Yaw, Knee, AnklePitch, AnkleRoll)
        1,  4,  6,  8, 10, 12,  // R leg
        2                       // WaistYaw
    };

    //////////////////////// ONNX Runtime ////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;

    int input_obs_idx_ = -1;
    int input_time_step_idx_ = -1;
    int output_actions_idx_ = -1;
    int output_joint_pos_idx_ = -1;
    int output_joint_vel_idx_ = -1;
    int output_body_pos_idx_ = -1;
    int output_body_quat_idx_ = -1;

    //////////////////////// Network Dimensions ////////////////////////
    // Tracking policy without state estimation (Tracking-Flat-P73-Wo-State-Estimation-v0):
    //   command(26) + motion_anchor_ori_b(6) + base_ang_vel(3)
    //   + joint_pos_rel(13) + joint_vel_rel(13) + last_action(13) = 74
    // (motion_anchor_pos_b and base_lin_vel are dropped — both require state estimation.)
    static constexpr int NUM_TRACKING_OBS = 74;
    static constexpr int NUM_TRACKING_ACT = 13;
    static constexpr int MOTION_NUM_BODIES_OUT = 8;   // body_names in cfg: base_link, WaistYaw, L_HipPitch, L_Knee, L_Foot, R_HipPitch, R_Knee, R_Foot
    static constexpr int MOTION_ANCHOR_IDX   = 0;     // base_link is first in body_names

    // Per-joint action scale (Isaac order), from ONNX metadata:
    //   0.25 * effort_limit / stiffness
    static constexpr std::array<double, 13> kActionScale = {
        0.057291666666666664, // [0] L_HipRoll:   0.25 * 352 / 1536
        0.057291666666666664, // [1] R_HipRoll
        0.06597222222222222,  // [2] WaistYaw:    0.25 * 152 / 576
        0.058666666666666666, // [3] L_HipPitch:  0.25 * 220 / 937.5
        0.058666666666666666, // [4] R_HipPitch
        0.038,                // [5] L_HipYaw:    0.25 * 95 / 625
        0.038,                // [6] R_HipYaw
        0.07357150286674972,  // [7] L_Knee:      0.25 * 220 / 747.552
        0.07357150286674972,  // [8] R_Knee
        0.04840125733055579,  // [9] L_AnklePitch: 0.25 * 95 / 490.644
        0.04840125733055579,  // [10] R_AnklePitch
        0.04845458176801621,  // [11] L_AnkleRoll: 0.25 * 95 / 490.104
        0.04845458176801621   // [12] R_AnkleRoll
    };

    //////////////////////// Observation / Action ////////////////////////
    std::vector<float> policy_frame_;   // 74D current obs frame

    // Raw policy output (Isaac order), fed back as `last_action` in next obs
    Matrix<double, NUM_TRACKING_ACT, 1> rl_action_;
    Matrix<double, NUM_TRACKING_ACT, 1> last_action_raw_;

    //////////////////////// Motion Cache (from ONNX aux outputs) ////////////////////////
    int total_frames_ = 0;
    int time_step_    = 0;
    Matrix<float, Dynamic, 13> motion_joint_pos_;       // [T, 13] Isaac order
    Matrix<float, Dynamic, 13> motion_joint_vel_;       // [T, 13] Isaac order
    Matrix<float, Dynamic, 3>  motion_anchor_pos_w_;    // [T, 3]  world frame, base_link
    Matrix<float, Dynamic, 4>  motion_anchor_quat_w_;   // [T, 4]  (w, x, y, z)

    //////////////////////// processNoise (unchanged) ////////////////////////
    bool is_on_robot_ = false;

    Matrix<double, MODEL_DOF, 1> q_noise_;
    Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Matrix<double, MODEL_DOF, 1> q_vel_noise_;
    Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    double noise_time_cur_ = 0.0;
    double noise_time_pre_ = 0.0;
    bool noise_initialized_ = false;

    static constexpr double lpf_cutoff_hz_ = 20.0;

    //////////////////////// Robot State ////////////////////////
    // Default joint positions in P73 order (13D) and Isaac order (13D).
    // Values come from ONNX metadata "default_joint_pos" in Isaac order:
    //   [0, 0, 0, 0.18, -0.18, 0, 0, 0.35, -0.35, -0.17, 0.17, 0, 0]
    Matrix<double, MODEL_DOF, 1> q_default_p73_;
    Matrix<double, 13, 1>        q_default_isaac_;

    // Joint position limits in P73 order (lower 12)
    Matrix<double, 12, 1> q_limit_lower_p73_, q_limit_upper_p73_;

    // PD gains in P73 order (13D)
    VectorQd kp_p73_, kd_p73_;
    VectorQd torque_bound_p73_;
    VectorQd torque_rl_;
    VectorQd torque_init_;
    VectorQd q_init_;
    VectorQd q_spline_;
    VectorQd torque_spline_;

    // 4-bar kinematics for sim motor-torque clamp
    FourBarKinematics sim_four_bar_;

    //////////////////////// Timing ////////////////////////
    float start_time_;
    float time_inference_pre_ = 0.0;
    bool cc_init_ = true;

    // Policy rate: 50 Hz
    double policy_dt_ = 0.02;

    // Motion pose "carry" — accumulated offset to make looping motion continue
    // smoothly forward in world space. At each loop boundary we add
    //   (motion_anchor_pos_w[last] - motion_anchor_pos_w[0])
    // to this offset, so anchor_pos_w used in the obs is continuous across loops.
    Vector3d motion_pos_offset_ = Vector3d::Zero();

    double value_ = 0.0;

    string weight_dir_;

    //////////////////////// Actuator Net (disabled for tracking) ////////////////////////
    bool use_actuator_net_ = false;

    struct ANetWeights {
        Eigen::Matrix<double, 32, 6>  W0;
        Eigen::Matrix<double, 32, 1>  b0;
        Eigen::Matrix<double, 32, 32> W1;
        Eigen::Matrix<double, 32, 1>  b1;
        Eigen::Matrix<double, 32, 32> W2;
        Eigen::Matrix<double, 32, 1>  b2;
        Eigen::Matrix<double, 1, 32>  W3;
        double b3;
    };
    std::array<ANetWeights, 12> anet_weights_;

    std::array<std::array<double, 2>, 12> anet_pos_err_hist_{};
    std::array<std::array<double, 2>, 12> anet_vel_hist_{};
    bool anet_hist_initialized_ = false;

    VectorQd cached_anet_torque_;

    static constexpr double anet_output_scale_ = 100.0;
    static constexpr double anet_dt_ = 0.01;

    void loadActuatorNets();
    void computeActuatorNetTorques();
    double anetForward(int joint_idx, const Eigen::Matrix<double, 6, 1>& input);

    //////////////////////// ROS2 Velocity Command (unused here but kept) ////////////////////////
    std::mutex vel_mutex_;
    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_yaw_ = 0.0;
    rclcpp::CallbackGroup::SharedPtr vel_cbg_;
    rclcpp::executors::SingleThreadedExecutor vel_executor_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
    std::thread vel_spin_thread_;
    std::atomic<bool> vel_spin_running_{false};

    void velCmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void startVelSubscriber();
    void stopVelSubscriber();

private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
};

#endif
