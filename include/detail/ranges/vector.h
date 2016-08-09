//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_VECTOR_GPU_H
#define GSTORM_VECTOR_GPU_H

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>

namespace gstorm {
  namespace range {

    template<typename T>
    struct _vector_gpu : public traits::range_forward_traits<T> {
    public:

      struct iterator : public std::random_access_iterator_tag {
        using range = _vector_gpu<T>;
        using difference_type = typename T::difference_type;
        using value_type = typename T::value_type;
        using reference = value_type&;
        using rvalue_reference = value_type&&;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        iterator(typename T::iterator it, T* __owner) : it(it), __owner(__owner) { }

        reference operator*() { return *it; }

        value_type operator*() const { return *it; }

        iterator& operator++() {
          ++it;
          return *this;
        }

        iterator operator++(int) {
          auto ip = it;
          ++it;
          return iterator(ip, __owner);
        }

        iterator& operator--() {
          --it;
          return *this;
        }

        iterator operator--(int) {
          auto ip = it;
          --it;
          return iterator(ip, __owner);
        }

        reference operator[](difference_type n) { return *(it + n); }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          return iterator(lhs.it + n, lhs.__owner);
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          return iterator(rhs.it + n, rhs.__owner);
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          return iterator(lhs.it - n, lhs.__owner);
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
      };

      using sentinel = iterator;
      using construction_type = std::tuple<T&>;

      _vector_gpu() = default;

      _vector_gpu(T& vec) : __owner(&vec) {}

      _vector_gpu(const construction_type& tpl) : __owner(&std::get<0>(tpl)) {}

      iterator begin() const { return iterator(std::begin(*__owner), __owner); }

      sentinel end() const { return sentinel(std::end(*__owner), __owner); }

    private:
      T* __owner;
    };

    template<typename T, typename A>
    auto vector(std::vector<T, A>& cont) {
      return _vector_gpu<std::vector<T, A>>(cont);
    }

    template<typename T>
    auto vector(std::tuple<T&>& cont) {
      return _vector_gpu<T>(cont);
    }

  }
}
#endif //GSTORM_VECTOR_GPU_H
