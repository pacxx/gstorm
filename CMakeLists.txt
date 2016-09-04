cmake_minimum_required(VERSION 3.2)
project(gstorm)

find_package(LLVM REQUIRED CONFIG)
find_package(CUDA REQUIRED)


add_compile_options(-std=pacxx -stdlib=libc++)
add_definitions(${LLVM_DEFINITIONS})

set(PACXX_PATH "/home/mhaidl/local" CACHE PATH "Path to PACXX")
set(PACXXRT_PATH "/home/mhaidl/agamemnon/code/pacxx_v2" CACHE PATH "Path to PACXX")
set(RANGES_V3_PATH "/home/mhaidl/code/range-v3" CACHE PATH "Path to PACXX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories("${PACXX_PATH}/kronos")
include_directories("${PACXXRT_PATH}/include")
include_directories("${RANGES_V3_PATH}/include")

link_directories("${PACXX_PATH}/lib")
link_directories("${PACXXRT_PATH}/build")

include_directories("./include")

set(SOURCE_FILES tests/transform.cpp include/meta/static_const.h include/detail/ranges/vector.h include/detail/traits.h include/meta/tuple_helper.h include/detail/operators/copy.h include/gstorm.h include/detail/actions/transform.h include/detail/operators/async.h include/detail/algorithms/transform.h include/detail/algorithms/reduce.h include/detail/decorators/ref.h include/detail/algorithms/for_each.h include/detail/algorithms/fill.h)
add_executable(gstorm ${SOURCE_FILES})