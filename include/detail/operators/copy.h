//
// Created by mhaidl on 09/08/16.
//

#ifndef GSTORM_COPY_H
#define GSTORM_COPY_H

#include <detail/traits.h>
#include <meta/static_const.h>
#include <PACXX.h>

namespace gstorm {
  namespace gpu {

    namespace detail {
      template<typename T>
      struct _gpu_copy {

        using value_type = typename T::value_type;

        _gpu_copy(const T& vec) : _dev_copy(pacxx::v2::get_executor().allocate<value_type>(vec.size())),
                                  _count(vec.size()) {
          _dev_copy.upload(vec.data(), vec.size());
        }


        operator T() {
          T ret(_count);

          _dev_copy.download(ret.data(), ret.size());

          return ret;
        }


      private:
        pacxx::v2::DeviceBuffer <value_type>& _dev_copy;
        size_t _count;
      };

    }

    struct _copy {
      template<typename T>
      auto operator()(const T& input) const {
        static_assert(traits::is_vector<T>::value && "Only std::vector is currently supported!");
        return detail::_gpu_copy<T>(input);
      }
    };

    template<typename T>
    auto operator|(const T& lhs, const _copy& rhs) {
      return rhs(lhs);
    }


    auto copy = gstorm::static_const<_copy>();

  }
}

#endif //GSTORM_COPY_H
