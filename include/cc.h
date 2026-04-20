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
    void processNoise();
    void processObservation();
    void feedforwardPolicy();
    Vector3d quatRotateInverse(const Quaterniond &q, const Vector3d &v);

    //////////////////////// Joint Order Permutation ////////////////////////
    // IsaacLab _LOWER_JOINT_NAMES order (12):
    //   L_HipRoll, L_HipPitch, L_HipYaw, L_Knee, L_AnklePitch, L_AnkleRoll,
    //   R_HipRoll, R_HipPitch, R_HipYaw, R_Knee, R_AnklePitch, R_AnkleRoll
    //
    // P73 JOINT_NAME order (13):
    //   L_HipYaw, L_HipRoll, L_HipPitch, L_Knee, L_AnklePitch, L_AnkleRoll,
    //   R_HipYaw, R_HipRoll, R_HipPitch, R_Knee, R_AnklePitch, R_AnkleRoll,
    //   WaistYaw
    //
    // kP73ToIsaac[p73_idx] = isaac_idx  (for building obs in IsaacLab order)
    // kIsaacToP73[isaac_idx] = p73_idx  (for applying actions to P73 joints)
    static constexpr std::array<int, 12> kP73ToIsaac = {
        2, 0, 1, 3, 4, 5,   // L leg
        8, 6, 7, 9, 10, 11  // R leg
    };
    static constexpr std::array<int, 12> kIsaacToP73 = {
        1, 2, 0, 3, 4, 5,   // L leg
        7, 8, 6, 9, 10, 11  // R leg
    };

    //////////////////////// ONNX Runtime ////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;

    int input_policy_idx_ = -1;
    int input_critic_idx_ = -1;
    int output_actions_idx_ = -1;
    int output_value_idx_ = -1;

    //////////////////////// Network Dimensions ////////////////////////
    // From rough_env_cfg.py PolicyCfg:
    //   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    //   + gait_phase_sin(1) + gait_phase_cos(1)
    //   + motor_joint_pos(12) + motor_joint_vel(12) + last_action(12) = 47
    static const int num_action = 12;         // lower body RL-controlled
    static const int num_single_obs = 47;

    int history_length_ = 10;     // overwritten from ONNX shape
    int policy_obs_dim_ = num_single_obs * 5;

    //////////////////////// Observation Buffers ////////////////////////
    std::vector<float> policy_frame_;                   // 47D single frame
    std::vector<float> policy_obs_hist_term_major_;     // 47*H term-major
    bool policy_hist_initialized_ = false;

    // Critic obs (if needed)
    std::vector<float> critic_obs_;

    //////////////////////// processNoise (TOCABI sim2real pattern) ////////////////////////
    // is_on_robot_: true=real robot (direct sensor + LPF), false=sim (noise + numerical diff + LPF)
    bool is_on_robot_ = true;

    // Filtered/noised joint state — used by BOTH obs and PD (matching TOCABI)
    Matrix<double, MODEL_DOF, 1> q_noise_;       // joint position (noised in sim, direct on robot)
    Matrix<double, MODEL_DOF, 1> q_noise_pre_;   // previous step (for numerical diff)
    Matrix<double, MODEL_DOF, 1> q_vel_noise_;   // joint velocity (numerical diff in sim, direct on robot)
    Matrix<double, MODEL_DOF, 1> q_dot_lpf_;     // LPF-filtered joint velocity (4Hz cutoff)

    double noise_time_cur_ = 0.0;
    double noise_time_pre_ = 0.0;
    bool noise_initialized_ = false;

    static constexpr double lpf_cutoff_hz_ = 20.0;  // TOCABI uses 4Hz LPF for q_dot

    //////////////////////// Robot State ////////////////////////
    // Default joint positions in P73 order (from p73_walker.py)
    Matrix<double, MODEL_DOF, 1> q_default_p73_;

    // Default joint positions in IsaacLab lower body order (12)
    Matrix<double, 12, 1> q_default_isaac_;

    // RL action (IsaacLab order, 12D)
    Matrix<double, num_action, 1> rl_action_;
    Matrix<double, num_action, 1> last_action_processed_;  // raw * scale

    // Joint position limits in P73 order (for q_des clamping, lower 12 only)
    Matrix<double, 12, 1> q_limit_lower_p73_, q_limit_upper_p73_;

    // PD gains in P73 order (13D, including WaistYaw)
    VectorQd kp_p73_, kd_p73_;
    VectorQd torque_bound_p73_;
    VectorQd torque_rl_;
    VectorQd torque_init_;
    VectorQd q_init_;
    VectorQd q_init_hold_;  // DEBUG: captured pose at mode entry
    VectorQd q_spline_;
    VectorQd torque_spline_;

    // 4-bar kinematics used in sim to reproduce motor-level torque clamping
    // through J^T (state_estimator does not populate rd_.four_bar_Jaco_ in simMode).
    FourBarKinematics sim_four_bar_;

    double action_scale_ = 0.5;  // from ActionsCfg scale

    //////////////////////// Timing ////////////////////////
    float start_time_;
    float time_inference_pre_ = 0.0;
    bool cc_init_ = true;

    // Policy rate: 50Hz (decimation=4, dt=0.005 → policy_dt=0.02)
    double policy_dt_ = 0.02;

    // Gait phase counter (50Hz step counter)
    int gait_step_counter_ = 0;
    int gait_period_steps_ = 60;  // from rough_env_cfg __post_init__

    // Velocity command (updated by ROS2 subscriber)
    std::mutex vel_mutex_;
    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_yaw_ = 0.0;
    double cmd_zero_max_ = 1.0e-3;

    double value_ = 0.0;

    string weight_dir_;

    //////////////////////// Actuator Net ////////////////////////
    // Per-joint neural network replacing PD control for lower body (12 joints).
    // Network: Linear(6,32)->Softsign->Linear(32,32)->Softsign->Linear(32,32)->Softsign->Linear(32,1)
    // Input: [pos_err_t, pos_err_{t-1}, pos_err_{t-2}, vel_t, vel_{t-1}, vel_{t-2}]
    // Output: motor current (A) -> x100 -> torque (Nm)
    // Computes at 50Hz (policy rate), cached for 1kHz main loop.
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

    // History buffers: [joint_idx][slot], slot: 0=~10ms ago, 1=~20ms ago
    // Updated at 100Hz (anet_dt_). Current values are computed fresh every tick.
    std::array<std::array<double, 2>, 12> anet_pos_err_hist_{};
    std::array<std::array<double, 2>, 12> anet_vel_hist_{};
    bool anet_hist_initialized_ = false;

    VectorQd cached_anet_torque_;

    static constexpr double anet_output_scale_ = 100.0;  // motor current (A) -> torque (Nm)
    static constexpr double anet_dt_ = 0.01;              // history update interval (10ms, 100Hz)

    void loadActuatorNets();
    void computeActuatorNetTorques();
    double anetForward(int joint_idx, const Eigen::Matrix<double, 6, 1>& input);

    //////////////////////// ROS2 Velocity Command Subscriber ////////////////////////
    // Uses dc_.node_ (main controller node) to share its DDS participant,
    // avoiding communication issues when running with sudo on real robot.
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
