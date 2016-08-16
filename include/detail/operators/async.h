//
// Created by mhaidl on 16/08/16.
//

#ifndef GSTORM_ASYNC_H
#define GSTORM_ASYNC_H

#include <meta/static_const.h>
#include <range/v3/all.hpp>
#include <vector>
#include <future>
#include <iostream>
#include <detail/ranges/vector.h>

namespace gstorm {
  namespace gpu {
    template<typename T>
    struct _async {
      using type_ = std::remove_reference_t<std::remove_cv_t<T>>;
      using value_type = decltype(*std::declval<type_>().begin());

      _async(T&& view) : _view(view) {}

      auto operator()() {

        // get a BindingPromise instance from PACXX
        // the PACXX runtime ensure that the promise is alive when the callback is fired
        auto& promise = pacxx::v2::get_executor().getPromise<range::gvector<std::vector<value_type>>>(
            ranges::v3::distance(_view));
        auto future = promise.getFuture(); // get an std::future from the promise
        auto& outRng = promise.getBoundObject(); // get the bound object that will survive until the callback fires

        gpu::algorithm::transform(_view, outRng, [](auto&& in) { return in; },
                                  [&]() mutable {
                                    promise.fulfill(); // at this point the computation is finished we can fulfill the promise
                                    pacxx::v2::get_executor().forgetPromise(promise);
                                  });

        return future;
      }


    private:
      T _view;
    };

    struct _async_helper {
      template<typename T>
      auto operator()(T&& view) { return _async<T>(view); }
    };

    auto async = gstorm::static_const<_async_helper>();

    template<typename View>
    auto operator|(View&& lhs, _async_helper&& rhs) {
      return rhs(lhs);
    };

  }
}

#endif //GSTORM_ASYNC_H
