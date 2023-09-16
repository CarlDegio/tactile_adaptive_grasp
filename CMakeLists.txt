cmake_minimum_required(VERSION 3.8)
project(tactile_adaptive_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(bye140_msg REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  script/adaptive_grasp.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  script/catch_tactile_video.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
