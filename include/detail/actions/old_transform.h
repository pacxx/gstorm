//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_GPU_TRANSFORM_H
#define GSTORM_GPU_TRANSFORM_H

#if !(defined(__CYGWIN__) && !defined(_WIN32)) // this is not pacxx disable GPU code

#include <tuple>
#include <utility>
#include <cstddef>
#include <type_traits>
#include <detail/ranges/vector.h>
#include <detail/operators/copy.h>
#include <meta/tuple_helper.h>
#include <meta/static_const.h>
#include <PACXX.h>

namespace gstorm {
  namespace gpu {


    template<typename T, typename... Ts>
    const T& __first(T&& arg, Ts&& ...) {
      return arg;
    }

    namespace detail {
      template<typename T>
      struct __grid_point {
        __grid_point(T x, T y, T z) : x(x), y(y), z(z) {};

        operator int() { return static_cast<int>(x); }

        operator std::ptrdiff_t() { return static_cast<long>(x); }

        operator std::pair<T, T>() { return {x, y}; }

        operator std::pair<std::ptrdiff_t, std::ptrdiff_t>() { return {x, y}; }

        T x, y, z;
      };

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"

      template<typename T, typename std::enable_if<!std::is_reference<
          typename T::range::value_type>::value>::type* = nullptr>
      typename T::reference saveDeRef(T&& iterator, bool enabled) {
        if (enabled)
          return *iterator;
        else {
          typename T::value_type dis;
          return dis;
        }
      }

      template<typename T, std::enable_if_t<std::is_reference<
          typename T::range::value_type>::value>* = nullptr>
      typename T::reference saveDeRef(T&& iterator, bool) {
        return *iterator;
      }

#pragma clang diagnostic pop

      template<typename T, typename U,
          std::enable_if_t<traits::is_vector<T>::value>* = nullptr>
      auto __decorate(U&& vec) {
        return range::gvector<T>(vec);
      }

      template<typename T, typename U,
          typename std::enable_if<!traits::is_vector<T>::value>::type* = nullptr>
      auto __decorate(U&& args) {
        return T(args);
      }
    }


    template<typename CType, typename... T, std::size_t... I>
    auto materialize_(const std::tuple<T...>& t, std::index_sequence<I...>) {
      return CType(std::get<I>(t)...);
    }

    template<typename CType, typename... T>
    auto materialize(const std::tuple<T...>& t) {
      return materialize_<CType>(t, std::make_index_sequence<sizeof...(T)>());
    }

    template<int Start, typename... T, std::size_t... I>
    auto subtuple_(const std::tuple<T...>& t, std::index_sequence<I...>) {
      return std::tie(std::get<Start + I>(t)...);
    }

    template<int Start, int Length, typename... T>
    auto subtuple(const std::tuple<T...>& t) {
      return subtuple_<Start>(t, std::make_index_sequence<Length>());
    }

    template<typename T>
    struct sizeofArgs : public sizeofArgs<decltype(&T::Create)> {
    };

    template<typename RType, typename... ArgTys>
    struct sizeofArgs<RType (*)(ArgTys...)> {
      enum {
        arity = sizeof...(ArgTys)
      };
    };

    template<size_t length, size_t pos>
    static constexpr size_t scan(const size_t arr[length], const size_t i = 0) {
      return (pos < length) ? ((i < pos) ? arr[i] + scan<length, pos>(arr, i + 1) : 0) : 0;
    }

    template<size_t length, size_t pos>
    static constexpr size_t getAt(const size_t arr[length]) {
      return (pos < length) ? arr[pos] : 0;
    }

    template<typename T>
    struct cond_decay {
      using type = std::conditional_t<traits::is_vector<std::remove_reference_t<T>>::value, T, std::decay_t<T>>;
    };

    namespace algorithm {
      template<typename OutputRng, typename... Args>
      struct transform_invoker {
        template<typename Func, typename... OArgs, typename... KArgs, size_t... I>
        void operator()(Func& func, std::tuple<OArgs...>& out,
                        std::tuple<KArgs...>& args, std::index_sequence<I...>) {

          auto arg_tuple = std::tuple_cat(out, args);
          auto k = pacxx::v2::kernel(
              [&func](OArgs... out, const KArgs... args) {
                auto g = Thread::get().global;
                detail::__grid_point<decltype(g.x)> __id(g.x, g.y, g.z);

                if (g.x >= __first(out...).end() - __first(out...).begin()) return;

                auto outIt =
                    detail::__decorate<typename OutputRng::iterator::range>(std::forward_as_tuple(out...)).begin() +
                    __id;

                constexpr size_t a[sizeof...(Args)] = {traits::view_traits<
                    std::remove_reference_t<typename std::remove_reference_t<Args>::iterator::range>>::arity...};

                auto targs = std::forward_as_tuple(args...);
                auto truncated =
                    [&](const auto& ... tuples) {
                      return subtuple<0, sizeof...(Args)>(
                          std::forward_as_tuple(tuples...));
                    }(subtuple<scan<sizeof...(Args), I>(a), getAt<sizeof...(Args), I>(a)>(
                        targs)...);

                *outIt = meta::apply(
                    [&](auto&& ... args) {
                      auto rngs = std::make_tuple(
                          (detail::__decorate<std::remove_reference_t<typename std::remove_reference_t<Args>::iterator::range>>(
                              args).begin() + __id)...);
                      return meta::apply([&](auto&& ... args) {
                        return func(std::forward_as_tuple(*args...));
                      }, rngs);
                    },
                    truncated);

              },
              {{static_cast<size_t>((std::get<0>(out).end() - std::get<0>(out).begin() + 127) / 128)},
               {128}});
          meta::apply(k, arg_tuple);
        }
      };


      struct _transform_algorithm {
        template<template<typename... Args> class InputRng,
            typename OutputRng, typename... Args, typename UnaryOp>
        auto operator()(InputRng<Args...>& inRng, OutputRng& outRng,
                        UnaryOp&& func) {
          auto input = meta::apply([](auto&& ... args) { return std::tuple_cat(args.begin().unwrap()...); },
                                   inRng._getRngs());
          auto output = outRng.begin().unwrap();
          transform_invoker<OutputRng, Args...> invoke;

          invoke(func, output, input,
                 std::make_index_sequence<std::tuple_size<decltype(input)>::value>{});
        }
      };


      auto transform = gstorm::static_const<_transform_algorithm>::value;

    }
    namespace action {
      template<typename T, typename F>
      struct _transform_action {
        _transform_action(T&& rng, F func) : _rng(rng), _func(func) {}

        auto operator()() {
          constexpr size_t thread_count = 128;
          using value_type = typename std::remove_reference_t<T>::value_type;
          auto kernel = pacxx::v2::kernel([](value_type* data, size_t size, F func) {
            auto id = Thread::get().global;

            if (static_cast<size_t>(id.x) >= size) return;

            data[id.x] = func(data[id.x]);

          }, {{(_rng.end() - _rng.begin() + thread_count - 1) / thread_count},
              {thread_count}});

          kernel(_rng.begin(), _rng.end() - _rng.begin(), _func);
        }

        operator typename std::remove_reference_t<T>::source_type() {
          __warning("_transform_action evaluating");
          operator()();
          __warning("_transform_action collapsing");
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

      template<typename Rng, typename F>
      auto operator|(data::_gpu_copy<Rng>&& lhs, const _transform_action_helper<F>& rhs) {
        return rhs(std::forward<decltype(lhs)>(lhs));
      }

      template<typename Rng, typename F>
      auto operator|(const Rng& lhs, const _transform_action_helper<F>& rhs) {
        auto trans = rhs(lhs |
                            gpu::copy); // evaluate explicitly here because _gpu_copy is destroyed befor it collapses into its source type
        return trans;
      };

    }
  }
}

#endif

#endif //GSTORM_GPU_TRANSFORM_H
