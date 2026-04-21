cmake_minimum_required(VERSION 3.10)
project(DigitalTwin2026)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add flags for better optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++17")

set(Ceres_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ceres-bin)
# Find Ceres package
find_package(Ceres REQUIRED)

# MuJoCo library path
set(MUJOCO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.3.7)

# Link directories
link_directories(${MUJOCO_DIR}/lib)

# Comoon include directories
# include_directories(
#     ${MUJOCO_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
#     ${CMAKE_CURRE_SOURCE_DIR}/simulate
#     /usr/include/eigen3                # Eigen headers
# )


##################################################################################
# Create executable for cyber physical system
add_executable(cyber_physical_system
            src/cyber_physical_system.cc 
            src/exp_meas_proc_func.cc 
            src/rs485_bus_functions.cc 
            src/my_mjc_func.cc 
            src/read_write_data_func.cc
)
# Include directories
target_include_directories(cyber_physical_system    PRIVATE
    ${MUJOCO_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate
    /usr/include/eigen3                # Eigen headers
)

# Enable AddressSanitizer in Debug mode
# target_compile_options(cyber_physical_system PRIVATE
#     $<$<CONFIG:Debug>:-fsanitize=address -fno-omit-frame-pointer -g>
# )

# target_link_options(cyber_physical_system PRIVATE
#     $<$<CONFIG:Debug>:-fsanitize=address>
# )

# Link libraries
target_link_libraries(cyber_physical_system PRIVATE
    mujoco
    glfw
    Ceres::ceres # Ceres library
    m     # Math library   
    pthread 
)

# Set output directory
set_target_properties(cyber_physical_system PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

##################################################################################
# Simulate executable
add_executable(simulate
    simulate_bioloid_setptctrl/main.cc
    simulate_bioloid_setptctrl/simulate.cc
    simulate_bioloid_setptctrl/glfw_adapter.cc
    simulate_bioloid_setptctrl/glfw_dispatch.cc
    simulate_bioloid_setptctrl/platform_ui_adapter.cc
    simulate_bioloid_setptctrl/lodepng.cpp
)

# Include directories
target_include_directories(simulate PRIVATE
                            ${MUJOCO_DIR}/include    
                            ${CMAKE_CURRENT_SOURCE_DIR}/simulate    
                            /usr/include/eigen3)

target_link_libraries(simulate PRIVATE
    mujoco
    glfw
    Ceres::ceres
    m
    pthread
)

set_target_properties(simulate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

##################################################################################
# # Create executable for Ceres trial
# add_executable(ceres_trial
#             src/ceres_trial.cc 
#             src/my_mjc_func.cc 
# )

# # Link libraries
# target_link_libraries(ceres_trial
#     mujoco
#     glfw
#     Ceres::ceres
#     pthread   # Threading support
#     m         # Math library
# )

# # Set output directory
# set_target_properties(ceres_trial PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
# )
