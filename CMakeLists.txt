cmake_minimum_required(VERSION 3.2)
project(gstorm)

find_package(LLVM REQUIRED CONFIG)
find_package(CUDA REQUIRED)




add_compile_options(-std=pacxx -stdlib=libc++)
add_definitions(${LLVM_DEFINITIONS})

set(PACXX_PATH "/home/mhaidl/local" CACHE PATH "Path to PACXX")
set(PACXXRT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../pacxx_v2" CACHE PATH "Path to PACXX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")



include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories("${PACXX_PATH}/kronos")
include_directories("${PACXXRT_PATH}/include")
include_directories("./include")

link_directories("${PACXX_PATH}/lib")
link_directories("${PACXXRT_PATH}/../build_pacxx")

set(SOURCE_FILES tests/transform.cpp include/view.h include/action.h include/meta/static_const.h include/detail/views/vector.h include/detail/actions/transform.h include/detail/views/zip.h include/detail/traits.h include/meta/tuple_helper.h include/detail/actions/gpu_transform.h)
add_executable(gstorm ${SOURCE_FILES})

target_link_libraries(gstorm pacxxrt2 pacxxhook c++ c++abi )