//
// Created by mhaidl on 10/08/16.
//

#ifndef GSTORM_TRANSFORM_H
#define GSTORM_TRANSFORM_H

#include <PACXX.h>
#include <range/v3/all.hpp>
#include <cstddef>
#include <type_traits>

namespace gstorm
{
  namespace gpu
  {
    namespace algorithm
    {

      template <typename InRng, typename OutRng, typename UnaryFunc>
      auto transform(InRng&& in, OutRng&& out, UnaryFunc&& func)
      {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel([](decltype(in.begin()) in,
                                           decltype(out.begin()) out,
                                           size_t distance,
                                           UnaryFunc func) {
          auto id = Thread::get().global;

          if (static_cast<size_t>(id.x) >= distance) return;

          out[id.x] = func(*(in + id.x));

        }, {{(distance + thread_count - 1) / thread_count},
            {thread_count}});

        kernel(in.begin(), out.begin(), distance, func);
      };

    }
  }
}

#endif //GSTORM_TRANSFORM_H
