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

template<typename InRng, typename OutRng, typename UnaryFunc>
auto transformGeneric(InRng &&in, OutRng &out, UnaryFunc &&func) {
  constexpr size_t thread_count = 128;

  int distance = ranges::v3::distance(in);

  auto inIt = in.begin();
  auto outIt = out.begin();

  auto functor = [=](auto &config) {
    auto id = config.get_global(0);
    if (id < distance)
      *(outIt + id) = func(*(inIt + id));
  };

  pacxx::v2::Executor::get().launch(functor, {{(distance + thread_count - 1) / thread_count},
                                              {thread_count}, 0});

}

template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
auto transformGeneric(InRng &&in,
                      OutRng &out, UnaryFunc &&func,
                      CallbackFunc &&callback) {
  constexpr size_t thread_count = 128;
  int distance = ranges::v3::distance(in);

  auto inIt = in.begin();
  auto outIt = out.begin();

  auto functor = [=](auto &config) {
    auto id = config.get_global(0);
    if (id < distance)
      *(outIt + id) = func(*(inIt + id));
  };

  pacxx::v2::Executor::get().launch_with_callback(functor,
                                                  {{(distance + thread_count - 1) / thread_count},
                                           {thread_count}, 0},
                                                  std::forward<CallbackFunc>(callback));
}

template<typename InRng, typename OutRng, typename UnaryFunc>
auto transform(InRng &&in, OutRng &out, UnaryFunc &&func) {

  return transformGeneric(std::forward<InRng>(in),
                          out,
                          std::forward<UnaryFunc>(func));

}

template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
auto transform(InRng &&in, OutRng &out, UnaryFunc &&func, CallbackFunc &&callback) {
  return transformGeneric(std::forward<InRng>(in),
                          out,
                          std::forward<UnaryFunc>(func),
                          callback);
}

}
}
}
