//
// Created by mhaidl on 23/08/16.
//

#pragma once

#include <cstddef>
#include <range/v3/all.hpp>
#include <PACXX.h>
#include <detail/ranges/vector.h>

namespace gstorm {
  namespace gpu {
    namespace algorithm {
      template<typename InTy, typename OutTy, typename BinaryFunc>
      struct reduce_functor {
        using value_type = decltype(*std::declval<InTy>());
      private:

        value_type gather(InTy it, size_t distance, BinaryFunc& func) const {
          const size_t stride = 64 * 128;
          value_type mine = *(it + Thread::get().global.x);

          for (size_t i = Thread::get().global.x + stride; i < distance; i += stride)
            mine = func(mine, *(it + i));

          return mine;
        }


        void reduce(value_type /*__attribute__((address_space(3)))*/* sdata, BinaryFunc& func) const {
          auto localId = Thread::get().index.x;
          for (size_t i = Block::get().range.x / 2; i > 32; i >>= 1) {
            if (localId < i)
              sdata[localId] = func(sdata[localId], sdata[localId + i]);

            Block::get().synchronize();
          }
        }

        void blockreduce(value_type /*__attribute__((address_space(3)))*/* sdata, OutTy out, BinaryFunc& func) const {
          auto localId = Thread::get().index.x;
          if (localId < 32) {
            sdata[localId] = func(sdata[localId], sdata[localId + 32]);
            Block::get().synchronize();
            sdata[localId] = func(sdata[localId], sdata[localId + 16]);
            Block::get().synchronize();
            sdata[localId] = func(sdata[localId], sdata[localId + 8]);
            Block::get().synchronize();
            sdata[localId] = func(sdata[localId], sdata[localId + 4]);
            Block::get().synchronize();
            sdata[localId] = func(sdata[localId], sdata[localId + 2]);
            Block::get().synchronize();
            sdata[localId] = func(sdata[localId], sdata[localId + 1]);
            Block::get().synchronize();
          }

          if (localId == 0)
            *out = sdata[0];
        }


      public:
        void operator()(InTy in, OutTy out, size_t distance, BinaryFunc func, value_type neutral) const {

          auto id = Thread::get();

          [[shared]] value_type sdata[128];
          sdata[id.index.x] = neutral;
          if (static_cast<size_t>(id.global.x) >= distance) return;

          sdata[id.index.x] = gather(in, distance, func);

          Block::get().synchronize();

          reduce(sdata, func);

          blockreduce(sdata, out + Block::get().index.x, func);
        }
      };

      template<typename InRng, typename BinaryFunc>
      auto reduce(InRng&& in, decltype(*in.begin()) neutral, BinaryFunc&& func) {
        constexpr size_t thread_count = 128;
        constexpr size_t block_count = 64;

        using value_type = decltype(*in.begin());

        auto distance = ranges::v3::distance(in);
        std::vector<value_type> result(block_count, neutral);
        range::gvector<std::vector<value_type>> out(result);

        auto kernel = pacxx::v2::kernel(
            reduce_functor<decltype(in.begin()), decltype(out.begin()), BinaryFunc>(),
            {{block_count},
             {thread_count}});

        kernel(in.begin(), out.begin(), distance, func, neutral);

        result = out;

        return std::accumulate(result.begin(), result.end(), neutral, func);

      };
    }
  }
}