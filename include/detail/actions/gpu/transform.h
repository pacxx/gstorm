//
// Created by mhaidl on 10/08/16.
//

#ifndef GSTORM_TRANSFORM_H
#define GSTORM_TRANSFORM_H

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

      template<typename InRng, typename OutRng, typename UnaryFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func) {
        constexpr size_t thread_count = 128;

        auto distance = ranges::v3::distance(in);

        auto kernel = pacxx::v2::kernel([](decltype(in.begin()) in,
                                           decltype(out.begin()) out,
                                           size_t distance,
                                           UnaryFunc func) {
          auto id = Thread::get().global;

          if (static_cast<size_t>(id.x) >= distance) return;

          *(out + id.x) = func(*(in + id.x));

        }, {{(distance + thread_count - 1) / thread_count},
            {thread_count}});

        kernel(in.begin(), out.begin(), distance, func);
      };


      template<typename InRng, typename OutRng, typename UnaryFunc, typename CallbackFunc>
      auto transform(InRng&& in, OutRng& out, UnaryFunc&& func, CallbackFunc&& callback) {
          constexpr size_t thread_count = 128;

          auto distance = ranges::v3::distance(in);

          auto kernel = pacxx::v2::kernel_with_cb([](decltype(in.begin()) in,
                                                     decltype(out.begin()) out,
                                                     size_t distance,
                                                     UnaryFunc func) {
            auto id = Thread::get().global;

            if (static_cast<size_t>(id.x) >= distance) return;

            *(out + id.x) = func(*(in + id.x));

          }, {{(distance + thread_count - 1) / thread_count},
              {thread_count}}, std::forward<CallbackFunc>(callback));

          kernel(in.begin(), out.begin(), distance, func);
      };

    }

    namespace action {
      template<typename T, typename F>
      struct _transform_action {
        _transform_action(T&& rng, F func) : _rng(std::forward<T>(rng)), _func(func) {}

        auto operator()() {
            algorithm::transform(_rng, _rng, _func);
        }

        operator typename std::remove_reference_t<T>::source_type() {
            operator()();
            return _rng;
        }

      private:
        T _rng;
        F _func;
      };

      template<typename F>
      struct _transform_action_helper {
        _transform_action_helper(F func) : _func(func) {}

        template<typename T>
        auto operator()(T&& rng) const {
            return _transform_action<T, F>(std::forward<T>(rng), _func);
        }

      private:
        F _func;
      };


      struct _transform {
        template<typename F>
        auto operator()(F func) const {
            return _transform_action_helper<decltype(func)>(func);
        }
      };

      auto transform = gstorm::static_const<_transform>();

      template<typename T, typename F>
      auto operator|(range::gvector<T>&& lhs, _transform_action_helper<F>&& rhs) {
          return rhs(std::forward<decltype(lhs)>(lhs));
      }

      template<typename Rng, typename F>
      auto operator|(Rng& lhs, _transform_action_helper<F>&& rhs) {
          // evaluate explicitly here because _gpu_copy is destroyed before it collapses into its source type
          auto gpu_copy = lhs | gpu::copy;
          // auto trans =
          return rhs(std::move(gpu_copy));;
      };

    }
  }
}
#endif //GSTORM_TRANSFORM_H
