cmake_minimum_required(VERSION 3.8)
project(p73_cc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Ofast)

# ONNX Runtime (bundled in thirdparty/)
set(ONNXRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(p73_lib REQUIRED)
find_package(p73_msgs REQUIRED)
find_package(br_driver REQUIRED)

include_directories(
  include
  ${p73_lib_INCLUDE_DIRS}
  /usr/local/include/br_driver
  ${ONNXRT_DIR}/include
)

add_library(${PROJECT_NAME}
  src/cc.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  geometry_msgs
  p73_lib
  p73_msgs
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRT_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
  br_driver
  ${ONNXRT_DIR}/lib/libonnxruntime.so
)

# Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

file(GLOB ONNXRT_LIBS "${ONNXRT_DIR}/lib/libonnxruntime*")
install(FILES ${ONNXRT_LIBS}
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY ${ONNXRT_DIR}/include/
  DESTINATION include
)

install(DIRECTORY policy/
  DESTINATION share/${PROJECT_NAME}/policy
)

# Export for find_package(p73_cc)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} onnxruntime)
ament_export_dependencies(rclcpp geometry_msgs p73_lib p73_msgs)

ament_package()
