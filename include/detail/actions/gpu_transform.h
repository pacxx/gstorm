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
#include <detail/views/vector.h>
#include <meta/tuple_helper.h>
#include <meta/static_const.h>
#include <PACXX.h>

namespace gstorm {
  namespace action {
    namespace gpu {


      template<typename T, typename... Ts>
      const T& __first(T&& arg, Ts&& ...) {
        return arg;
      }

      namespace detail {
        template<typename T>
        struct __grid_point {
          __grid_point(T x, T y, T z) : x(x), y(y), z(z) { };

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
          return view::_vector_view<T>(vec);
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
                          (detail::__decorate<std::remove_reference_t<typename std::remove_reference_t<Args>::iterator::range>>(args).begin() + __id)...);
                      return meta::apply([&](auto&&... args) {
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


    struct _transform {
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


      auto transform = gstorm::static_const<_transform>::value;

    }
  }
}

#endif

#endif //GSTORM_GPU_TRANSFORM_H
