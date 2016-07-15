//
// Created by mhaidl on 05/07/16.
//

#ifndef GSTORM_ZIP_H
#define GSTORM_ZIP_H

#include <tuple>
#include <iterator>
#include <vector>
#include <detail/traits.h>
#include <meta/tuple_helper.h>
#include <iostream>

namespace gstorm {
  namespace view {

    template<typename... T>
    struct _zip_view {
      using indices = std::make_index_sequence<sizeof...(T)>;

    public:
      template<typename TIt>
      struct iterator : public std::random_access_iterator_tag {
        using range = _zip_view<T...>;
        using difference_type = std::ptrdiff_t;
        using value_type = std::tuple<typename std::remove_reference_t<T>::value_type...>;
        using reference = std::tuple<typename std::remove_reference_t<T>::reference...>;
        using rvalue_reference = value_type&&;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        iterator(TIt it) : it(it) { }

        reference operator*() { return getElements(indices()); }

        value_type operator*() const { return getElements(indices()); }

        iterator& operator++() {
          meta::for_each_in_tuple([](auto& it) { ++it; }, it);
          return *this;
        }

        iterator operator++(int) {
          auto ip = it;
          meta::for_each_in_tuple([](auto& it) { ++it; }, it);
          return iterator(ip);
        }

        iterator& operator--() {
          meta::for_each_in_tuple([](auto& it) { --it; }, it);
          return *this;
        }

        iterator operator--(int) {
          auto ip = it;
          meta::for_each_in_tuple([](auto& it) { --it; }, it);
          return iterator(ip);
        }

        reference operator[](difference_type n) { return *(it + n); }

        friend iterator operator+(const iterator& lhs, difference_type n) {
          auto ip = lhs.it;
          meta::for_each_in_tuple([&](auto& it) { it += n; }, ip);
          return iterator(ip);
        }

        friend iterator operator+(difference_type n, const iterator& rhs) {
          auto ip = rhs.it;
          meta::for_each_in_tuple([&](auto& it) { it += n; }, ip);
          return iterator(ip);
        }

        friend iterator operator-(const iterator& lhs, difference_type n) {
          auto ip = lhs.it;
          meta::for_each_in_tuple([&](auto& it) { it -= n; }, ip);
          return iterator(ip);
        }

        friend difference_type operator-(const iterator& left, const iterator& right) {
          return std::get<0>(left.it) - std::get<0>(right.it);
        }

        friend iterator& operator-=(iterator& lhs, difference_type n) {
          meta::for_each_in_tuple([&](auto& it) { it -= n; }, lhs.it);
          return lhs;
        }

        friend iterator& operator+=(iterator& lhs, difference_type n) {
          meta::for_each_in_tuple([&](auto& it) { it += n; }, lhs.it);
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

        auto unwrap() { return getUnwraped(indices()); }

      private:

        template<size_t ...Is>
        auto getUnwraped(std::index_sequence<Is...>) const {
          return std::tuple_cat(std::get<Is>(it).unwrap()...);
        }

        template<size_t ...Is>
        auto getElements(std::index_sequence<Is...>) {
          return std::tie(*std::get<Is>(it)...);
        }


        template<size_t ...Is>
        auto getElements(std::index_sequence<Is...>) const {
          return std::make_tuple(*std::get<Is>(it)...);
        }

        TIt it;
      };

      template<typename TIt>
      using sentinel = iterator<TIt>;

      using construction_type = std::tuple<T& ...>;

      _zip_view() = default;

      _zip_view(T&& ... rngs) : _rngs(std::make_tuple(rngs...)) { }

      auto begin() {
        auto its = getBegin(indices());
        return iterator<decltype(its)>(its);
      }

      auto end() {
        auto its = getEnd(indices());
        return sentinel<decltype(its)>(its);
      }

      auto _getRngs() { return _rngs; }

    private:


      template<size_t ...Is>
      auto getBegin(std::index_sequence<Is...>) {
        return std::make_tuple(std::get<Is>(_rngs).begin()...);
      }

      template<size_t ...Is>
      auto getEnd(std::index_sequence<Is...>) {
        return std::make_tuple(std::get<Is>(_rngs).end()...);
      }

      std::tuple<T...> _rngs;
    };

    template<typename T, std::enable_if_t<traits::is_vector<std::remove_reference_t<T>>::value>* = nullptr>
    auto __decorate(T&& vec) {
      return _vector_view<std::remove_reference_t<T>>(vec);
    }

    template<typename T, std::enable_if_t<!traits::is_vector<std::remove_reference_t<T>>::value>* = nullptr>
    auto __decorate(T&& vec) {
      return vec;
    }


    template<typename... T>
    auto zip(T&& ... rng) {
      return _zip_view<decltype(__decorate(std::forward<T>(rng)))...>(__decorate(std::forward<T>(rng))...);
    }


  }
}
#endif //GSTORM_ZIP_H
