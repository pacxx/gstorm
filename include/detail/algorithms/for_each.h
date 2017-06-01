//
// Created by mhaidl on 27/08/16.
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

#include <detail/common/Meta.h>

namespace gstorm {
  namespace gpu {
    namespace algorithm {

      template<typename InRng, typename UnaryFunc>
      auto for_each(InRng&& in, UnaryFunc&& func) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto inIt = in.begin();

        auto for_each_functor = [=](auto &config) {
          auto id = config.get_global(0);
            if (static_cast<size_t>(id) >= distance) return;
          func(*(inIt + id));
        };

        auto kernel = pacxx::v2::kernel(
            for_each_functor,
            {{(distance + thread_count - 1) / thread_count},
             {thread_count}, 0, 0});

        kernel();
      };
    }
  }
}
