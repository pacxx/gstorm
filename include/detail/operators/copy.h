//
// Created by mhaidl on 09/08/16.
//

#ifndef GSTORM_COPY_H
#define GSTORM_COPY_H

#include <detail/traits.h>
#include <meta/static_const.h>
#include <detail/ranges/vector.h>

#include <PACXX.h>

namespace gstorm {
  namespace gpu {

    struct _copy {
      template<typename T>
      auto operator()(T& input) const {
          static_assert(traits::is_vector<T>::value, "Only std::vector is currently supported!");
          return range::gvector<T>(input);
      }
    };

    template<typename T>
    auto operator|(T& lhs, const _copy& rhs) {
      return rhs(lhs);
    }


    auto copy = gstorm::static_const<_copy>();

  }
}

#endif //GSTORM_COPY_H
