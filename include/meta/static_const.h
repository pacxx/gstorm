//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_STATIC_CONST_H
#define GSTORM_STATIC_CONST_H
namespace gstorm {
  template<typename T>
  struct static_const {
    static constexpr T value{};
  };

  template<typename T>
  constexpr T static_const<T>::value;
}
#endif //GSTORM_STATIC_CONST_H
