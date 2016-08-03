//
// Created by Michael on 21.07.2016.
//

#ifndef GSTORM_SCALAR_H
#define GSTORM_SCALAR_H

#include <tuple>
#include <iterator>
#include <vector>
#include <limits>
#include <detail/traits.h>

namespace gstorm {
  namespace view {

    template<typename T>
    struct _scalar_view  {
    public:
        using storage_type = std::conditional_t<std::is_reference<T>::value, std::remove_reference_t<T>*, T>;
        using base_type = T;
        using value_type = T;
        using reference = const value_type&;
        using size_type = size_t;

        using difference_type = std::ptrdiff_t;

      struct iterator : public std::random_access_iterator_tag {
        using range = _scalar_view<T>;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using reference = const value_type&;
        using rvalue_reference = value_type&&;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        iterator(storage_type value) : _value(value) { }

        template<typename U = storage_type, std::enable_if_t<std::is_pointer<U>::value>* = nullptr> 
        const auto& operator*() const { return *_value; }
        template<typename U = storage_type, std::enable_if_t<!std::is_pointer<U>::value>* = nullptr> 
        auto operator*() const { return _value; }

        iterator& operator++() {
            return *this;
        }

        iterator operator++(int) {
          return iterator(_value);
        }

        iterator& operator--() {
          return *this;
        }

        iterator operator--(int) {
          return iterator(_value);
        }

        reference operator[](difference_type n) { return _value; }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          return iterator(lhs._value);
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          return iterator(rhs._value);
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          return iterator(lhs._value);
        }

        friend difference_type operator-(const iterator& left, const iterator& right) {
          return std::numeric_limits<difference_type>::max();
        }

        friend iterator& operator-=(iterator& lhs, difference_type n) {
          return lhs;
        }

        friend iterator& operator+=(iterator& lhs, difference_type n) {
          return lhs;
        }

        friend bool operator<(const iterator& left, const iterator& right) {
          return true;
        }

        friend bool operator>(const iterator& left, const iterator& right) {
          return true;
        }

        friend bool operator<=(const iterator& left, const iterator& right) {
          return true;
        }

        friend bool operator>=(const iterator& left, const iterator& right) {
          return false;
        }

        bool operator==(const iterator& other) const { return true; }

        bool operator!=(const iterator& other) const { return true; }

        template<typename U = storage_type, std::enable_if_t<!std::is_pointer<U>::value>* = nullptr> 
        auto unwrap() { return std::tuple<T>(_value); }
        template<typename U = storage_type, std::enable_if_t<std::is_pointer<U>::value>* = nullptr> 
        auto unwrap() { return std::tuple<const T&>(*_value); }
        

      private:
        storage_type _value;
      };

      using sentinel = iterator;
      using construction_type = std::tuple<T>;

        _scalar_view() = default;

        template<typename U = storage_type, std::enable_if_t<!std::is_pointer<U>::value>* = nullptr> 
        _scalar_view(const T& value) : _value(value) { }
        template<typename U = storage_type, std::enable_if_t<std::is_pointer<U>::value>* = nullptr> 
        _scalar_view(const T& value) : _value(&value) { }

        _scalar_view(const construction_type& tpl) : _scalar_view(std::get<0>(tpl)) { }

      iterator begin() { return iterator(_value); }

      sentinel end() { return sentinel(_value); }

    private:
      storage_type _value;
    };

    template<typename T>
    auto scalar(const T value) {
      return _scalar_view<T>(value);
    }

    template<typename T>
    auto scalar(std::tuple<T>& cont) {
      return _scalar_view<T>(cont);
    }

  }
}

#endif //GSTORM_SCALAR_H
