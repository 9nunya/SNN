%%writefile CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(snn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
include_directories(include)

file(GLOB SRC_FILES "src/*.cpp")
file(GLOB CU_SRC_FILES "src/*.cu")

# 1. Force CMake to treat these .cpp files completely as CUDA files
set_source_files_properties(src/neuron.cpp src/synapse.cpp PROPERTIES LANGUAGE CUDA)

add_executable(snn ${SRC_FILES} ${CU_SRC_FILES})

# 2. MANDATORY: Enable relocatable device code linking on the executable target
set_target_properties(snn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(snn PRIVATE CUDA::cudart)