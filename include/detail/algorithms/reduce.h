//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <cstddef>
#include <range/v3/all.hpp>
#include <PACXX.h>
#include <detail/ranges/vector.h>

#define REDUCE_THREAD_N 128ul
#define REDUCE_BLOCK_N 130

namespace gstorm {

  namespace meta {
    template<int First, int Last, typename Fn>
    struct _static_for {
      _static_for(Fn f) : func(f) {}

      template<typename T>
      auto operator()(T init, T y) const {
        return _static_for<First + 1, Last, Fn>(func)(init, func(init, y));
      }

      Fn func;
    };

    template<int N, typename Fn>
    struct _static_for<N, N, Fn> {
      _static_for(Fn f) {}

      template<typename T>
      auto operator()(T init, T y) const {
        return y;
      }
    };

    template<int First, int Last, typename Fn>
    auto static_for(Fn f) {
      return _static_for<First, Last, Fn>(f);
    }

  }


  namespace gpu {
    namespace algorithm {
      template<typename InTy, typename OutTy, typename BinaryFunc>
      struct reduce_functor {
        using value_type = std::remove_reference_t<decltype(*std::declval<InTy>())>;
      private:
        BinaryFunc func;
      public:

        reduce_functor(BinaryFunc&& f) : func(f) {}

        void operator()(InTy in, OutTy out, size_t distance, value_type init, size_t ept) const {

          [[shared]] value_type sdata[REDUCE_THREAD_N];

          auto block = Block::get();
          size_t tid = Thread::get().index.x;
          auto n = pacxx::v2::_stage([&] { return distance; });
          auto IsPow2 = (n & (n - 1)) == 0;

          int elements_per_thread = pacxx::v2::_stage([&] { return ept; });

          value_type sum = init;

          size_t gridSize = REDUCE_THREAD_N * Grid::get().range.x;
          size_t i = Thread::get().global.x;

          for (int x = 0; x < elements_per_thread; ++x) {
            sum = func(sum, *(in + i));
            i += gridSize;
          }

          if (!IsPow2)
            while (i < n) {
              sum = func(sum, *(in + i));
              i += gridSize;
            }

          sdata[tid] = sum;
          block.synchronize();

          if (tid < 64)
            sdata[tid] = func(sdata[tid], sdata[tid + 64]);
          block.synchronize();
          if (tid < 32) {
            volatile value_type* sm = &sdata[0];
            sm[tid] = func(sm[tid], sm[tid + 32]);
            sm[tid] = func(sm[tid], sm[tid + 16]);
            sm[tid] = func(sm[tid], sm[tid + 8]);
            sm[tid] = func(sm[tid], sm[tid + 4]);
            sm[tid] = func(sm[tid], sm[tid + 2]);
            sm[tid] = func(sm[tid], sm[tid + 1]);
          }
          if (tid == 0)
            *(out + static_cast<size_t>(block.index.x)) = sdata[tid];
        }
      };


      template<typename InRng, typename BinaryFunc>
      auto reduce(InRng&& in, std::remove_reference_t<decltype(*in.begin())> init, BinaryFunc&& func) {
        size_t distance = ranges::v3::distance(in);
        size_t thread_count = std::min(REDUCE_THREAD_N, distance);
        size_t ept = 1;
        if (distance > thread_count * 2) {
          do {
            ept *= 2;
          }
          while (distance / (thread_count * ept) > 130);
        }

        __error(ept);
        size_t block_count = std::max(distance / (thread_count * ept), 1ul);

        using value_type = std::remove_reference_t<decltype(*in.begin())>;
        std::vector<value_type> result(block_count, init);
        range::gvector<std::vector<value_type>> out(result);

        auto kernel = pacxx::v2::kernel(
            reduce_functor<decltype(in.begin()), decltype(out.begin()), BinaryFunc>(std::forward<BinaryFunc>(func)),
            {{block_count},
             {thread_count}});

        kernel(in.begin(), out.begin(), distance, init, ept);


        result = out;

        return std::accumulate(result.begin(), result.end(), init, func);
      };
    }
  }
}