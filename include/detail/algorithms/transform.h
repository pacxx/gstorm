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

      template<typename InTy, typename OutTy, typename UnaryFunc>
      struct transform_functor {
        void operator()(InTy in,
                        OutTy out,
                        size_t distance,
                        UnaryFunc func) const {
          auto id = get_global_id(0);
          if (static_cast<size_t>(id) >= distance) return;

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
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func, config cfg) {

        auto kernel = pacxx::v2::kernel(
            transform_functor<decltype(in.begin()), decltype(out.begin()), UnaryFunc>(),
            {{cfg.blocks.x, cfg.blocks.y, cfg.blocks.z},
             {cfg.threads.x, cfg.threads.y, cfg.threads.z}});

        kernel(in.begin(), out.begin(), func);
      };


      template<typename InRng, typename OutRng, typename UnaryFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel(
            transform_functor<decltype(in.begin()), decltype(out.begin()), UnaryFunc>(),
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}, 0});

        kernel(in.begin(), out.begin(), distance, func);
      };

      template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func, CallbackFunc&& callback) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel_with_cb(
            transform_functor<decltype(in.begin()), decltype(out.begin()), UnaryFunc>(),
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}, 0}, std::forward<CallbackFunc>(callback));

        kernel(in.begin(), out.begin(), distance, func);
      };
    }
  }
}
