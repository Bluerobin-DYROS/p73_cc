#ifndef P73_CC_H
#define P73_CC_H


//--- Standard Libraries
#include <filesystem>
#include <fstream>
#include <map>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>

//--- RobotData
#include "p73_lib/robot_data.h"
#include "p73_lib/p73.h"
#include "wholebody_functions.h"

//--- ROS2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"

//--- ONNX Runtime
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <unordered_map>
#include <ament_index_cpp/get_package_share_directory.hpp>

class CustomController  
{
public:
    CustomController(DataContainer &dc, RobotEigenData &rd);
    ~CustomController();
    void copyRobotData(RobotEigenData &rd_global_);


    void computeFast(); // for general case, use this 

    static constexpr int num_action = 12;

private:
    DataContainer &dc_;
    RobotEigenData rd_cc_; // local copy of RobotEigenData for CC computations
    RobotEigenData &rd_;

    void initVariable();
    void loadOnnX(const std::string& model_path);

    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    
    //--- ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::unordered_map<std::string, std::unique_ptr<Ort::Session>> sessions_;

    //--- Setting
    double hz_ = 50.0;  // Default inference frequency, can be overridden by config
    bool debug_enabled_ = true;  // Enable/disable debug output
    int debug_print_frequency_ = 100;  // Print debug info every N iterations
    // bool is_on_robot_ = true;
    bool is_on_robot_ = false;
    // std::string onnx_path = "2026-03-30_17-43-08_model_3200.onnx";
    std::string onnx_path = "2026-04-06_09-33-35_model_10000.onnx";
    int input_obs_idx_ = -1;
    int output_action_idx_ = -1;

    bool cc_init_ = true;
    bool cc_mode_active_prev_ = false;

    Eigen::VectorQd q_min;
    Eigen::VectorQd q_max;

    //--- RL Input
    // static constexpr int num_cur_state = 55;
    static constexpr int num_cur_state = 51;
    static constexpr int num_cur_h = 256;
    std::vector<float> state_cur_, h_cur_;
    std::vector<float> normalized_state_cur_;

    Eigen::VectorQd q_init_;
    
    Eigen::VectorQd q_noise_;
    Eigen::VectorQd q_noise_pre_;
    Eigen::VectorQd q_vel_noise_;

    Eigen::VectorQd torque_init_;
    Eigen::VectorQd torque_spline_;
    Eigen::VectorQd torque_rl_;

    //--- Motion / command states
    Eigen::Vector3d base_lin_vel_;
    Eigen::Vector3d base_ang_vel_;
    Eigen::Vector3d commands_;

    // Walking/stepping related variables
    double step_ticks_ = 0.0;
    int phase_indicator_ = 0;

    // Phase tracking
    double step_period_s_ = 0.0;
    double phase_period_s_ = 0.0;
    double phase_time_s_ = 0.0;
    double phase_time_s_init_ = 0.0;
    double phase_offset_s_ = 0.0;
    bool phase_started_ = true;

    // Command and control variables
    double target_heading_ = 0.0;
    bool heading_mode_ = false;

    // Stride limits
    double max_stride_x = 0.5;
    double max_stride_y = 0.3;
    double max_stride_yaw = 0.5;

    // Joystick velocity scaling
    double vel_scale_x_ = 0.3;
    double vel_scale_y_ = 0.3;
    double vel_scale_yaw_ = 0.3;
        
    // Time tracking
    double del_t = 0.0;
    double control_time_s = 0.0;
    double control_time_us = 0.0;
    double start_time_us = 0.0;
    double time_cur_s = 0.0;
    double time_pre_s = 0.0;
    double time_init_s = 0.0;
    double time_inference_pre_us = 0.0;

    //--- RL Functions
    void processNoise();
    void processObservation();
    void feedforwardPolicy();

    //--- RL Output
    Eigen::VectorXd rl_action_;

    // ROS2 subscribers
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;

    // ONNX RUNTIME
    Ort::Session* getSession(const std::string& name);
    size_t computeElementCount(const std::vector<int64_t>& shape);

    // Actor network (main network)
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;
};

#endif