//
// Created by mhaidl on 05/08/16.
//

#ifndef GSTORM_VIEW_TRANSFORM_H
#define GSTORM_VIEW_TRANSFORM_H

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>
#include "materialize.h"

namespace gstorm {
  namespace view {

    template<typename T, typename F>
    struct _transform_view : public traits::range_forward_traits<T> {
    public:

      struct iterator : public std::random_access_iterator_tag {
        using range = _transform_view<T, F>;
        using difference_type = typename T::difference_type;
        using value_type = typename T::value_type;
        using reference = value_type&;
        using rvalue_reference = value_type&&;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        iterator(typename T::iterator it, T* __owner, F func) : it(it), __owner(__owner), _func(func) {}

        value_type operator*() const { return _func(*it); }

        iterator& operator++() {
          ++it;
          return *this;
        }

        iterator operator++(int) {
          auto ip = it;
          ++it;
          return iterator(ip, __owner, _func);
        }

        iterator& operator--() {
          --it;
          return *this;
        }

        iterator operator--(int) {
          auto ip = it;
          --it;
          return iterator(ip, __owner, _func);
        }

        reference operator[](difference_type n) { return *(it + n); }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          return iterator(lhs.it + n, lhs.__owner, lhs._func);
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          return iterator(rhs.it + n, rhs.__owner, rhs._func);
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          return iterator(lhs.it - n, lhs.__owner, lhs._func);
        }

        friend difference_type operator-(const iterator& left, const iterator& right) {
          return left.it - right.it;
        }

        friend iterator& operator-=(iterator& lhs, difference_type n) {
          lhs.it -= n;
          return lhs;
        }

        friend iterator& operator+=(iterator& lhs, difference_type n) {
          lhs.it += n;
          return lhs;
        }

        friend bool operator<(const iterator& left, const iterator& right) {
          return left.it < right.it;
        }

        friend bool operator>(const iterator& left, const iterator& right) {
          return left.it > right.it;
        }

        friend bool operator<=(const iterator& left, const iterator& right) {
          return left.it <= right.it;
        }

        friend bool operator>=(const iterator& left, const iterator& right) {
          return left.it >= right.it;
        }

        bool operator==(const iterator& other) const { return other.it == it; }

        bool operator!=(const iterator& other) const { return other.it != it;; }

        auto unwrap() { return std::tie(*__owner); }

      private:
        typename T::iterator it;
        T* __owner;
        F _func;
      };

      using sentinel = iterator;
      using construction_type = std::tuple<T&>;

      _transform_view() = default;

      _transform_view(T& vec, F func) : __owner(vec), _func(func) {}

      _transform_view(const construction_type& tpl) : __owner(std::get<0>(tpl)) {}

      iterator begin() { return iterator(std::begin(__owner), &__owner, _func); }

      sentinel end() { return sentinel(std::end(__owner), &__owner, _func); }


      template<typename ToType>
      operator ToType() {
        return materialize.transform<ToType>(*this, [](auto in) { return in; });
      }


    private:
      T __owner;
      F _func;
    };

    template<typename Rng, typename F>
    auto transform(Rng& cont, F func) {
      return _transform_view<Rng, F>(cont, func);
    }

    template<typename T, typename F>
    auto transform(std::tuple<T&>& cont, F func) {
      return _transform_view<T, F>(cont, func);
    }

  }
}
#endif //GSTORM_TRANSFORM_H
