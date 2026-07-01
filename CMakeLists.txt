cmake_minimum_required(VERSION 3.18)
project(snn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
include_directories(include include/compute)

file(GLOB SRC_FILES "src/*.cpp" "src/compute/*.cpp")
file(GLOB CU_SRC_FILES "src/*.cu" "src/compute/*.cu")

# These .cpp files need CUDA compilation because they use __host__ __device__
set_source_files_properties(
    src/neuron.cpp
    src/synapse.cpp
    src/simulation.cpp
    src/compute/compute_ops.cpp
    src/compute/cpu.cpp
    src/cppn.cpp
    src/brain.cpp
    src/tournament_recorder.cpp
    PROPERTIES LANGUAGE CUDA
)

add_executable(snn ${SRC_FILES} ${CU_SRC_FILES})

set_target_properties(snn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(snn PRIVATE CUDA::cudart)
