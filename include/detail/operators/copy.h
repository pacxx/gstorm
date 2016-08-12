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

    namespace data {
      template<typename T>
      struct _gpu_copy {

        using value_type = typename T::value_type;
        using source_type = T;

        _gpu_copy(const T& vec) : _dev_copy(pacxx::v2::get_executor().allocate<value_type>(vec.size())),
                                  _count(vec.size()) {
          _dev_copy.upload(vec.data(), vec.size());
        }

        ~_gpu_copy(){
          _dev_copy.abandon();
          __error("dead");
        }

        operator T() {
          T ret(_count);
          __warning("_gpu_copy collapsing");
          _dev_copy.download(ret.data(), ret.size());

          return ret;
        }

        auto begin() { return _dev_copy.get(); }

        auto end() { return _dev_copy.get() + _count; }


      private:
        pacxx::v2::DeviceBuffer <value_type>& _dev_copy;
        size_t _count;
      };

    }

    struct _copy {
      template<typename T>
      auto operator()(T& input) const {
        static_assert(traits::is_vector<T>::value && "Only std::vector is currently supported!");
        return range::gpu_vector(input);
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
