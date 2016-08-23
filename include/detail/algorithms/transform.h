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

namespace gstorm {
  namespace gpu {
    namespace algorithm {

      template<typename InTy, typename OutTy, typename UnaryFunc>
      struct transform_functor {
        void operator()(InTy in,
                        OutTy out,
                        size_t distance,
                        UnaryFunc func) const {
          auto id = Thread::get().global;
          if (static_cast<size_t>(id.x) >= distance) return;

          *(out + id.x) = func(*(in + id.x));
        }
      };

      template<typename InRng, typename OutRng, typename UnaryFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel(
            transform_functor<decltype(in.begin()), decltype(out.begin()), UnaryFunc>(),
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}});

        kernel(in.begin(), out.begin(), distance, func);
      };

      template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func, CallbackFunc&& callback) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel_with_cb(
            transform_functor<decltype(in.begin()), decltype(out.begin()), UnaryFunc>(),
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}}, std::forward<CallbackFunc>(callback));

        kernel(in.begin(), out.begin(), distance, func);
      };
    }
  }
}
