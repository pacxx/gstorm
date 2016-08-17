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

  namespace traits {
    template<typename T>
    struct range_forward_traits {
      using base_type = T;

      using size_type = typename T::size_type;
      using value_type = typename T::value_type;
      using reference = std::conditional_t<std::is_const<T>::value, const typename T::reference, typename T::reference>;
      using const_reference = typename T::const_reference;
      using difference_type = typename T::difference_type;
    };
  }

  namespace range {

    template<typename T>
    struct gvector : public traits::range_forward_traits<T> {
    public:
      using source_type = T;
      using size_type = typename T::size_type;
      using value_type = typename T::value_type;
      using reference = std::conditional_t<std::is_const<T>::value, const typename T::reference, typename T::reference>;
      using const_reference = typename T::const_reference;
      using difference_type = typename T::difference_type;

      struct iterator : public ranges::v3::random_access_iterator_tag, public traits::range_forward_traits<T> {
        using size_type = typename T::size_type;
        using value_type = typename T::value_type;
        using reference = std::conditional_t<std::is_const<T>::value, const typename T::reference, typename T::reference>;
        using const_reference = typename T::const_reference;
        using difference_type = typename T::difference_type;
        using iterator_category = ranges::v3::random_access_iterator_tag;

        iterator() = default;

        explicit iterator(typename T::value_type* it) : it(it) {}

        reference operator*() { return *it; }

        reference operator*() const { return *it; }

        iterator& operator++() {
          ++it;
          return *this;
        }

        iterator operator++(int) const {
          auto ip = it;
          ++it;
          return iterator(ip);
        }

        iterator& operator--() {
          --it;
          return *this;
        }

        iterator operator--(int) const {
          auto ip = it;
          --it;
          return iterator(ip);
        }

        reference operator[](difference_type n) { return *(it + n); }

        friend iterator advance(const iterator& lhs, difference_type n) {
          return lhs.advance(n);
        }

        iterator advance(difference_type n) const {
          it += n;
          return *this;
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
        mutable typename T::value_type* it;
      };

      using sentinel = iterator;

      gvector() : _buffer(nullptr), _size(0) {}

      gvector(size_t size) : _buffer(&pacxx::v2::get_executor().allocate<typename T::value_type>(size)),
                             _size(size) {
//        __message("allocated ", (void*)_buffer);
      }

      gvector(const T& vec) : gvector(vec.size()) {
        _buffer->upload(vec.data(), vec.size());
      }

      ~gvector() {
//        __message("destroyed ", (void*)_buffer);
        if (_buffer)
          _buffer->abandon();
      }

      gvector(const gvector&) = delete;

      gvector(gvector&& other) {
        _buffer = other._buffer;
        //     __message("moved ", (void*)_buffer);
        other._buffer = nullptr;
        _size = other._size;
        other._size = 0;
      }

      gvector& operator=(const gvector&) = delete;

      gvector& operator=(gvector&&) = delete;

      iterator begin() noexcept { return iterator(_buffer->get()); }

      iterator end() noexcept { return iterator(_buffer->get(_size)); }

      const iterator begin() const noexcept { return iterator(_buffer->get()); }

      const iterator end() const noexcept { return iterator(_buffer->get(_size)); }

      operator std::remove_cv_t<T>() const {
        T tmp(_size);
        _buffer->download(tmp.data(), tmp.size());
        return tmp;
      }

    private:
      pacxx::v2::DeviceBuffer<typename T::value_type>* _buffer;
      size_t _size;
    };

    template<typename T>
    auto gpu_vector(T& vec) {
      return gvector<T>(vec);
    }
  }
}
#endif //GSTORM_VECTOR_GPU_H
