cmake_minimum_required(VERSION 3.2)
project(gstorm)

if(!WIN32)
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

link_directories("${PACXX_PATH}/lib")
link_directories("${PACXXRT_PATH}/../build_pacxx")

target_link_libraries(gstorm pacxxrt2 pacxxhook c++ c++abi )
else(!WIN32)
    add_compile_options(-std=c++14)

endif(!WIN32)

include_directories("./include")

set(SOURCE_FILES tests/transform.cpp include/view.h include/action.h include/meta/static_const.h include/detail/ranges/vector.h include/detail/actions/transform.h include/detail/views/zip.h include/detail/traits.h include/meta/tuple_helper.h include/detail/actions/gpu_transform.h include/detail/views/repeat.h include/detail/views/materialize.h include/detail/views/transform.h include/detail/operators/copy.h)
add_executable(gstorm ${SOURCE_FILES})