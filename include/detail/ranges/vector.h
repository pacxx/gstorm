//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_VECTOR_GPU_H
#define GSTORM_VECTOR_GPU_H

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>
#include <type_traits>
#include <PACXX.h>
#include <range/v3/view_facade.hpp>

namespace gstorm {
  namespace range {

    template<typename T>
    struct gvector : public traits::range_forward_traits<T> {
    public:

      struct iterator : public ranges::v3::random_access_iterator_tag {
        using difference_type = typename T::difference_type;
        using value_type = typename T::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using rvalue_reference = value_type&&;
        using iterator_category = ranges::v3::random_access_iterator_tag;

        iterator() = default;

        explicit iterator(typename T::value_type* it) : it(it) {}

     //   reference operator*() { return *it; }

        const_reference operator*() const { return *it; }

        iterator& operator++() {
          ++it;
          return *this;
        }

        iterator operator++(int) {
          auto ip = it;
          ++it;
          return iterator(ip);
        }

        iterator& operator--() {
          --it;
          return *this;
        }

        iterator operator--(int) {
          auto ip = it;
          --it;
          return iterator(ip);
        }

        reference operator[](difference_type n) { return *(it + n); }

        friend iterator advance(const iterator& lhs, difference_type n) {
          return iterator(lhs.it + n);
        }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          return iterator(lhs.it + n);
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          return iterator(rhs.it + n);
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          return iterator(lhs.it - n);
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

        bool operator!=(const iterator& other) const { return other.it != it; }

      private:
        typename T::value_type* it;
      };

      using sentinel = iterator;

      gvector() : _ptr(nullptr), _size(0) { }

      gvector(T& vec) : _size(vec.size()) {
#ifndef __device_code__
        auto& buffer = pacxx::v2::get_executor().allocate<typename T::value_type>(vec.size());
        _ptr = buffer.get();
        buffer.upload(vec.data(), vec.size());
#endif
      }

      ~gvector(){
#ifndef __device_code__
        if (_ptr) {
          auto buffer = pacxx::v2::get_executor().rt().translateMemory(_ptr);
          buffer->abandon();
        }
#endif
      }

      gvector(const gvector&) = default;

      gvector(gvector&& other)
      {
        _ptr = other._ptr;
        other._ptr = nullptr;
        _size = other._size;
        other._size = 0;
      }
      gvector& operator=(const gvector&) = delete;
      gvector& operator=(gvector&&) = delete;

      iterator begin() noexcept { return iterator(_ptr); }
      iterator end() noexcept { return iterator(_ptr + _size); }

      const iterator begin() const noexcept { return iterator(_ptr); }
      const iterator end() const noexcept { return iterator(_ptr + _size); }

      operator T(){
        T tmp(_size);
#ifndef __device_code__
        auto buffer = pacxx::v2::get_executor().rt().translateMemory(_ptr);
        buffer->download(tmp.data(), tmp.size());
#endif
        return tmp;
      }

    private:
      typename T::value_type* _ptr;
      size_t _size;
    };

    template<typename T>
    auto gpu_vector(T& vec) {
      return gvector<T>(vec);
    }
  }
}
#endif //GSTORM_VECTOR_GPU_H
