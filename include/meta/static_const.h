//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_STATIC_CONST_H
#define GSTORM_STATIC_CONST_H

#include <utility>

namespace gstorm {
  template<typename T>
  struct static_const {
    static constexpr T value{};

    template<typename... Ts>
    auto operator()(Ts&& ... args) {
      return value(std::forward<Ts>(args)...);
    }

    operator T() { return value; }

  };

  template<typename T>
  constexpr T static_const<T>::value;
}
#endif //GSTORM_STATIC_CONST_H
