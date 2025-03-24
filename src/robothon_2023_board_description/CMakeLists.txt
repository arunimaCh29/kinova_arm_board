cmake_minimum_required(VERSION 3.0.0)
project(robothon_2023_board_description)

find_package(catkin REQUIRED COMPONENTS roslaunch urdf xacro)

catkin_package(
  CATKIN_DEPENDS urdf xacro
)

include_directories(${catkin_INCLUDE_DIRS})

install(
  DIRECTORY launch meshes urdf robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
