// Copyright (c) 2019-2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace nda {

  // lazy expression for mapping a function on arrays A
  template <typename F, typename... A>
  struct expr_call;

  // impl details
  template <typename... Char>
  constexpr char _impl_find_common_algebra(char x0, Char... x) {
    return (((x == x0) && ...) ? x0 : 'N');
  }

  // algebra
  template <typename F, typename... A>
  constexpr char get_algebra<expr_call<F, A...>> = _impl_find_common_algebra(get_algebra<std::decay_t<A>>...);

  // NdArray concept
  template <typename F, typename... A>
  inline constexpr bool is_ndarray_v<expr_call<F, A...>> = true;

  //----------------------------

  template <class F>
  struct mapped;

  // a lazy expression of f(A...) where f is the function to map and A the arrays
  // e.g. f is sqrt(x) and there is one A, or f is min(x,y) and there are 2 A
  template <typename F, typename... A>
  struct expr_call {

    F f;
    std::tuple<const A...> a; // tuple of array (ref) on which f is applied

    private: // FIXME C++20 lambda implementation details
    template <size_t... Is, typename... Args>
    [[gnu::always_inline]] [[nodiscard]] auto _call(std::index_sequence<Is...>, Args const &... args) const {
      // a priori, we just need to return
      // f(std::get<Is>(a)(args...)...)
      // but if a(args) is a view, we can not always use f.
      // It works if f is well written, i.e. it takes a temporary and forwards it, like
      // f(auto &&x) { return ...}
      // but if the user's function is badly written, like (auto const & x) { ...}
      // which is quite natural for a simple function, then the temporary view will be dangling.
      // So we take care of this case explicitly
      if constexpr (is_regular_or_view_v<decltype(std::get<0>(a)(args...))>) { // we have views, same for all a...
        return mapped<F>{f}(std::get<Is>(a)(args...)...);
      } else {
        return f(std::get<Is>(a)(args...)...);
      }
    }
    template <size_t... Is, typename Args>
    [[gnu::always_inline]] auto _call_bra(std::index_sequence<Is...>, Args const &args) const {
      return f(std::get<Is>(a)[args]...);
    }

    public:
    template <typename... Args>
    auto operator()(Args const &... args) const {
      return _call(std::make_index_sequence<sizeof...(A)>{}, args...);
    }

    // vector interface
    template <typename Args>
    auto operator[](Args &&args) const {
      return _call_bra(std::make_index_sequence<sizeof...(A)>{}, args);
    }

    // FIXME copy needed for the && case only. Overload ?
    [[nodiscard]] auto shape() const { return std::get<0>(a).shape(); }

    [[nodiscard]] long size() const { return a.size(); }
  };

  /*
   * The lambda which maps function F onto the array
   */
  template <class F>
  struct mapped {
    F f;

    template <typename A0, typename... A>
    expr_call<F, A0, A...> operator()(A0 &&a0, A &&... a) const {
      EXPECTS(((a.shape() == a0.shape()) && ...)); // same shape
      return {f, {std::forward<A0>(a0), std::forward<A>(a)...}};
    }
  };

  /**
  * 
  * Maps a function onto the array (elementwise)
  *
  * @tparam F A lambda (do not use function pointers here, make a small lambda it is easier)
  * @param f : function to be mapped
  * @return a lambda that accepts array(s) as argument and return a lazy call expressions.
  *
  * @example nda_map.cpp
  */
  template <class F>
  mapped<F> map(F f) {
    return {std::move(f)};
  }

} // namespace nda
