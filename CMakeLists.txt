cmake_minimum_required(VERSION 3.8)
project(p73_cc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-Ofast)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(p73_lib REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# ONNX Runtime (installed via install_onnx.sh)
set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "/usr/local/lib")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_LIB_DIR})

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime library not found. Please run install_onnx.sh first.")
endif()

message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Build br_rl library
add_library(${PROJECT_NAME} SHARED
  src/cc.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  p73_lib
  ament_index_cpp
  sensor_msgs
)

target_link_libraries(${PROJECT_NAME}
  ${ONNXRUNTIME_LIB}
)

# Install library
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install ONNX models
install(DIRECTORY onnx/
  DESTINATION share/${PROJECT_NAME}/onnx
)

# Export dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include ${ONNXRUNTIME_INCLUDE_DIR})
ament_export_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
ament_export_dependencies(
  rclcpp
  p73_lib
  ament_index_cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()