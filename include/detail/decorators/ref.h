//
// Created by mhaidl on 27/08/16.
//

#pragma once

#include <detail/ranges/vector.h>

namespace gstorm {
  namespace gpu {
    template<typename T>
    struct _gref {
      using base_type = typename range::gvector<T>;
      using iterator = typename base_type::iterator;
      using sentinel = typename base_type::sentinel;
      using reference = typename base_type::reference;
      using difference_type = typename base_type::difference_type;
    private:
      iterator it;
      sentinel sen;
    public:

      _gref(base_type& ref) : it(ref.begin()), sen(ref.end()) {}

      iterator begin() const { return it; }

      sentinel end() const { return sen; }

      reference operator[](difference_type n) const {
        return *(it + n);
      }

      difference_type size() const { return sen - it; }

    };

    template<typename T>
    auto ref(range::gvector <T>& ref) {
      return _gref<T>(ref);
    }
  }
}