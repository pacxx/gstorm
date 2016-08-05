//
// Created by mhaidl on 05/08/16.
//

#ifndef GSTORM_MATERIALIZE_H
#define GSTORM_MATERIALIZE_H

#include <detail/traits.h>
#include <type_traits>
#include <detail/traits.h>
#include <action.h>

namespace gstorm {
  struct _materializer {
    template<typename T, typename ViewT, typename F>
    T transform(ViewT& rng, F func) const {
      static_assert(traits::is_vector<T>::value && "Only std::vector is currently supported!");

      T container(rng.end() - rng.begin());

      action::transform(rng, container, func);

      return container;

    }
  };

  auto materialize = gstorm::static_const<_materializer>::value;
}


#endif //GSTORM_MATERIALIZE_H
