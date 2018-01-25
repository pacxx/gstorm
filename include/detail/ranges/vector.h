//
// Created by mhaidl on 05/07/16.
//

#pragma once

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>
#include <type_traits>
#include <PACXX.h>
#include <range/v3/view_facade.hpp>
#include <detail/algorithms/fill.h>

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
      using pointer = value_type*;

      struct iterator : public std::random_access_iterator_tag, public traits::range_forward_traits<T> {
        using size_type = typename T::size_type;
        using value_type = typename T::value_type;
        using reference = std::conditional_t<std::is_const<T>::value, const typename T::reference, typename T::reference>;
        using const_reference = typename T::const_reference;
        using difference_type = typename T::difference_type;
        using iterator_category = std::random_access_iterator_tag;
        using pointer = value_type*; 

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
      }

      gvector(size_t size, value_type value) : _buffer(
          &pacxx::v2::get_executor().allocate<typename T::value_type>(size)),
                                               _size(size) {
        gpu::algorithm::fill(*this, value);
      }

      gvector(T& vec) : _buffer(&pacxx::v2::get_executor().allocate<typename T::value_type>(vec.size())),
                              _size(vec.size()) {
        _buffer->upload(vec.data(), vec.size());
      }

      ~gvector() {
//        __message("destroyed ", (void*)_buffer);
        if (_buffer)
          pacxx::v2::get_executor().free(*_buffer); 
      }

      gvector(const gvector& src) = delete;

      gvector(gvector&& other) {
        _buffer = other._buffer;
        other._buffer = nullptr;
        _size = other._size;
        other._size = 0;
      }

      gvector& operator=(const gvector& src) {
          src._buffer->copyTo(_buffer->get());
        return *this;
      }

      gvector& operator=(gvector&& other) {
        _buffer = other._buffer;
        other._buffer = nullptr;
        _size = other._size;
        other._size = 0;
        return *this;
      }

      iterator begin() noexcept { return iterator(_buffer->get()); }

      iterator end() noexcept { return iterator(_buffer->get(_size)); }

      const iterator begin() const noexcept { return iterator(_buffer->get()); }

      const iterator end() const noexcept { return iterator(_buffer->get(_size)); }

      auto size() const { return _size; }

      void resize(size_type size) {
        auto new_buffer = &pacxx::v2::get_executor().allocate<typename T::value_type>(size);
        if (_buffer)
          _buffer->copyTo(new_buffer->get());
        _size = size;
        if (_buffer)
          pacxx::v2::get_executor().free(*_buffer); 
        _buffer = new_buffer;
      }

      operator std::remove_cv_t<T>() const {
        T tmp(_size);
        _buffer->download(tmp.data(), tmp.size());
        return tmp;
      }

      void swap(gvector& other) {
        std::swap(_buffer, other._buffer);
        std::swap(_size, other._size);
      }

      typename T::value_type* data() {
        return _buffer->get();
      }

      const typename T::value_type* data() const {
        return _buffer->get();
      }

      reference operator[](difference_type n) const {
        return *_buffer->get(n);
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
