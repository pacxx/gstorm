//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_TRANSFORM_H
#define GSTORM_TRANSFORM_H
#include <meta/static_const.h>

namespace gstorm {
  namespace action {

    struct _transform {
      template <typename InRng, typename OutRng>
      auto operator()(InRng&& in, OutRng&& out){
        auto OIt = out.begin();
        auto It = in.begin();
        while (It != in.end())
          *(OIt)++ = *(It++);
      }


      template <typename InRng, typename OutRng, typename UnaryFn>
      auto operator()(InRng&& in, OutRng&& out, UnaryFn&& fn){
        auto OIt = out.begin();
        auto It = in.begin();
        while (It != in.end())
          *(OIt)++ = fn(*(It++));
      }

    };

    auto transform = gstorm::static_const<_transform>::value;

  }
}
#endif //GSTORM_TRANSFORM_H
