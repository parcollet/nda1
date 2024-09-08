// Copyright (c) 2019-2023 Simons Foundation
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
//
// Authors: Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides functionality to evaluate lazy expressions from the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./operation.hpp"
#include "./placeholder.hpp"
#include "./function.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_eval
   * @{
   */
  // ----------   eval  : forward decl  -------------

  template <typename T>
  FORCEINLINE decltype(auto) eval(T &&x, auto &&...pairs);

  // -----------------------

  template <int N, typename... Pairs>
  FORCEINLINE decltype(auto) eval_impl(placeholder<N>, Pairs &...pairs) {

    // Position of the pair which contains N or -1
    constexpr int N_position = []<size_t... Ps>(std::index_sequence<Ps...>) {
      return ((Pairs::p == N ? int(Ps) + 1 : 0) + ...) - 1;
    }(std::make_index_sequence<sizeof...(Pairs)>{});

    if constexpr (N_position == -1) { // N is not one of the Is
      return placeholder<N>{};
    } else {                                                   // N is one of the Is
      auto &pair_N = std::get<N_position>(std::tie(pairs...)); // FIXME pairs...[N_position]
      // the pair is a temporary constructed for the time of the eval call
      // if it holds a reference, we return it, else we move the rhs object out of the pair
      if constexpr (std::is_lvalue_reference_v<decltype(pair_N.rhs)>)
        return pair_N.rhs;
      else
        return std::move(pair_N.rhs);
    }
  }

  // -----------------------

  template <typename Tag, typename... Childs>
  decltype(auto) eval_impl(expr<Tag, Childs...> const &ex, auto &...pairs) {

    return [&]<size_t... Is>(std::index_sequence<Is...>)
#if defined(__GNUC__) and not defined(__clang__)
       mutable __attribute__((always_inline)) // For some reason clang and gcc need the mutable and attribute in different order ?
#else
       __attribute__((always_inline)) mutable
#endif
          ->decltype(auto) {
      if constexpr ((is_lazy<decltype(eval(std::get<Is>(ex.childs), pairs...))> or ...))
        return expr{Tag{}, eval(std::get<Is>(ex.childs), pairs...)...};
      else
        return operation<Tag>{}(eval(std::get<Is>(ex.childs), pairs...)...);
    }
    (std::make_index_sequence<sizeof...(Childs)>{});
  }
  // -----------------------

  template <typename T>
  FORCEINLINE decltype(auto) eval_impl(std::reference_wrapper<T> const &wrapper, auto &...pairs) {
    return eval(wrapper.get(), pairs...);
  }

  //Evaluates the underlying expression and rebuild the function.
  template <typename T, int... Is, typename... Pairs>
  FORCEINLINE decltype(auto) eval_impl(make_fun_impl<T, Is...> const &f, Pairs &...pairs) {
    // makes no sense if some of the Pairs placeholders are included in the Is.
    constexpr uint64_t I = ((1ull << Is) + ...);
    constexpr uint64_t J = ((1ull << Pairs::p) + ...);
    static_assert((I & J) == 0, "Impossible evaluation. You can not evaluate a make_fun_impl on the placeholders used to define the function");
    return make_function(eval(f.obj, pairs...), placeholder<Is>{}...);
  }

  // -----------------------

  /**
   * @brief Evaluate expression on pairs (placeholder = value)
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto res = nda::clef::eval(ex, i_ = 1, j_ = 2); // int res = 3;
   * @endcode
   *
   * If x is 
   *    a clef::expr, replace the placeholders by their value and recompute the expression.
   *        if all the placeholders are specified (full evaluation), return the result. 
   *        Otherwise (partial evaluation), return another clef::expr<> with only the remaining placeholders.
   *    a make_fun_impl, evaluate the underlying expression and reconstructs a new function from the expression.
   *      FIXME: EXAMPLE
   *    anything else, pass x through.
   *
   * @tparam T Any type
   * @tparam Pairs Must be nda::clef::pair<...>
   * @param x Expression/object to be evaluated. 
   * @param pairs of (placeholder, value) 
   * @return Cf below.
   */
  template <typename T>
  FORCEINLINE decltype(auto) eval(T &&x, auto &&...pairs) {
    if constexpr (requires { eval_impl(std::forward<T>(x), pairs...); })
      return eval_impl(std::forward<T>(x), pairs...);
    else
      return std::forward<T>(x);
  }
  /** @} */

} // namespace nda::clef
