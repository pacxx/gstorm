//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <cstddef>
#include <range/v3/all.hpp>
#include <PACXX.h>
#include <detail/ranges/vector.h>

#define REDUCE_THREAD_N 128
#define REDUCE_BLOCK_N 130

namespace gstorm {
  namespace gpu {
    namespace algorithm {
      template<typename InTy, typename OutTy, typename BinaryFunc>
      struct reduce_functor {
        using value_type = std::remove_reference_t<decltype(*std::declval<InTy>())>;
      private:

      public:
        void operator()(InTy in, OutTy out, size_t n, BinaryFunc func, value_type init) const {

      [[shared]] value_type sdata[REDUCE_THREAD_N]; 

      volatile value_type *sm = &sdata[0];
  
      auto block = Block::get();
      auto tid = Thread::get().index.x;
      auto nIsPow2 = (n & (n - 1)) == 0;
  
      value_type sum = init;
      auto gridSize = REDUCE_THREAD_N * 2 * Grid::get().range.x;
      auto i = block.index.x * REDUCE_THREAD_N * 2 + tid;
      while (i < n) {
        sum = func(sum, *(in + i));
        
        if (nIsPow2 || i + REDUCE_THREAD_N < n)
          sum = func(sum, *(in + i + REDUCE_THREAD_N));
        
        i += gridSize;
      }
  
      sm[tid] = sum;
      block.synchronize();
  
        if (tid < 64)
          sm[tid] = func(sm[tid], sm[tid + 64]);
        block.synchronize();
      if (tid < 32) {
          sm[tid] = func(sm[tid], sm[tid + 32]);
          sm[tid] = func(sm[tid], sm[tid + 16]);
          sm[tid] = func(sm[tid], sm[tid + 8]);
          sm[tid] = func(sm[tid], sm[tid + 4]);
          sm[tid] = func(sm[tid], sm[tid + 2]);
          sm[tid] = func(sm[tid], sm[tid + 1]);
      }
      if (tid == 0)
        *(out + block.index.x) = sm[tid];
        }
      };

      template<typename InRng, typename BinaryFunc>
      auto reduce(InRng&& in, std::remove_reference_t<decltype(*in.begin())> init, BinaryFunc&& func) {
        constexpr size_t thread_count = REDUCE_THREAD_N;
        constexpr size_t block_count = REDUCE_BLOCK_N;

        using value_type = std::remove_reference_t< decltype(*in.begin())>;

        auto distance = ranges::v3::distance(in);
        std::vector<value_type> result(block_count, init);
        range::gvector<std::vector<value_type>> out(result);

        auto kernel = pacxx::v2::kernel(
            reduce_functor<decltype(in.begin()), decltype(out.begin()), BinaryFunc>(),
            {{block_count},
             {thread_count}});

        kernel(in.begin(), out.begin(), distance, func, init);

        result = out;

        return std::accumulate(result.begin(), result.end(), init, func);

      };
    }
  }
}
