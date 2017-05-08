//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <PACXX.h>
#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <detail/operators/copy.h>
#include <detail/ranges/vector.h>
#include <meta/static_const.h>
#include <detail/algorithms/config.h>

namespace gstorm {
namespace gpu {
namespace algorithm {
namespace detail {
template<typename InTy, typename OutTy, typename UnaryFunc>
struct transform_functorGPU {
  void operator()(InTy in,
                  OutTy out,
                  size_t distance,
                  UnaryFunc func) const {
    auto id = get_global_id(0);
    if (static_cast<size_t>(id) >= distance)
      return;

    *(out + id) = func(*(in + id));
  }
  void operator()(InTy in,
                  OutTy out,
                  UnaryFunc func) const {
    auto id = get_global_id(0) + get_global_id(1) * get_grid_size(0);

    *(out + id) = func(*(in + id));
  }

};

template<typename InTy, typename OutTy, typename UnaryFunc>
struct transform_functorCPU {
  void operator()(InTy in,
                  OutTy out,
                  size_t distance,
                  UnaryFunc func) const {
    auto id = get_global_id(0);
    if (static_cast<size_t>(id) >= distance)
      return;

    *(out + id) = func(*(in + id));
  }
  void operator()(InTy in,
                  OutTy out,
                  UnaryFunc func) const {
    auto id = get_global_id(0) + get_global_id(1) * get_grid_size(0);

    *(out + id) = func(*(in + id));
  }

};

template<typename InRng, typename OutRng, typename UnaryFunc>
auto transformGPUNvidia(InRng &&in, OutRng &out, UnaryFunc &&func) {
  constexpr size_t thread_count = 128;

  auto distance = ranges::v3::distance(in);

  using FunctorTy = transform_functorGPU<decltype(in.begin()), decltype(out.begin()), UnaryFunc>;

  auto kernel = pacxx::v2::kernel<FunctorTy, pacxx::v2::Target::GPU>(
      FunctorTy(),
      {{(distance + thread_count - 1) / thread_count},
       {thread_count}, 0});

  kernel(in.begin(), out.begin(), distance, func);
}

template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
auto transformGPUNvidia(InRng &&in, OutRng &out, UnaryFunc &&func, CallbackFunc &&callback) {
  constexpr size_t thread_count = 128;

  auto distance = ranges::v3::distance(in);

  using FunctorTy = transform_functorGPU<decltype(in.begin()), decltype(out.begin()), UnaryFunc>;

  auto kernel = pacxx::v2::kernel_with_cb<FunctorTy, CallbackFunc, pacxx::v2::Target::GPU>(
      FunctorTy(),
      {{(distance + thread_count - 1) / thread_count},
       {thread_count}, 0}, std::forward<CallbackFunc>(callback));

  kernel(in.begin(), out.begin(), distance, func);
}

template<typename InRng, typename OutRng, typename UnaryFunc>
auto transformCPU(InRng &&in, OutRng &out, UnaryFunc &&func) {
  constexpr size_t thread_count = 128;

  auto distance = ranges::v3::distance(in);

  using FunctorTy = transform_functorCPU<decltype(in.begin()), decltype(out.begin()), UnaryFunc>;

  auto kernel = pacxx::v2::kernel<FunctorTy, pacxx::v2::Target::GPU>(
      FunctorTy(),
      {{(distance + thread_count - 1) / thread_count},
       {thread_count}, 0});

  kernel(in.begin(), out.begin(), distance, func);
}

template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
auto transformCPU(InRng &&in, OutRng &out, UnaryFunc &&func, CallbackFunc &&callback) {
  constexpr size_t thread_count = 128;

  auto distance = ranges::v3::distance(in);

  using FunctorTy = transform_functorCPU<decltype(in.begin()), decltype(out.begin()), UnaryFunc>;

  auto kernel = pacxx::v2::kernel_with_cb<FunctorTy, CallbackFunc, pacxx::v2::Target::GPU>(
      FunctorTy(),
      {{(distance + thread_count - 1) / thread_count},
       {thread_count}, 0}, std::forward<CallbackFunc>(callback));

  kernel(in.begin(), out.begin(), distance, func);
}
}

template<typename InRng, typename OutRng, typename UnaryFunc>
auto transform(InRng &&in, OutRng &out, UnaryFunc &&func) {
  using namespace pacxx::v2;
  auto &exec = Executor::get(0); // get default executor

  switch (exec.getExecutingDeviceType()) {
  case ExecutingDevice::GPUNvidia:
    return detail::transformGPUNvidia(std::forward<InRng>(in),
                                      out,
                                      std::forward<UnaryFunc>(func));
  case ExecutingDevice::CPU:return detail::transformCPU(std::forward<InRng>(in), out,
                                                        std::forward<UnaryFunc>(func));
  }
}

template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
auto transform(InRng &&in, OutRng &out, UnaryFunc &&func, CallbackFunc &&callback) {
  using namespace pacxx::v2;
  auto &exec = Executor::get(0); // get default executor

  switch (exec.getExecutingDeviceType()) {
  case ExecutingDevice::GPUNvidia:
    return detail::transformGPUNvidia(std::forward<InRng>(in),
                                      out,
                                      std::forward<UnaryFunc>(func),
                                      callback);
  case ExecutingDevice::CPU:
    return detail::transformCPU(std::forward<InRng>(in),
                                out,
                                std::forward<UnaryFunc>(func),
                                callback);
  }
}

}
}
}
