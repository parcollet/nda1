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
 * @brief Provides functionality to turn lazy expressions into callable objects and to generally simplify the evaluation
 * of lazy expressions.
 */

#pragma once
#include <type_traits>
#include <utility>

#include "./utils.hpp"
#include "./placeholder.hpp"
#include "../macros.hpp"

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Make a function out of a lazy expression
   *
   * @details Given a lazy expression, this class adapts it as a function of its placeholder.
   *           Its arguments are the values of placeholders used in evaluating the expression,
   *              in the order given by PlaceholderIndex.
   *
   * The following example shows how to turn a binary lazy expression `ex` into a callable object `f` that takes two
   * arguments:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto f = nda::clef::function{ex, i_, j_}; // this is a function (i, j)-> eval(ex, i_=i, j_=j)
   * auto res = f(1, 2);    // int res = 3;
   * auto res2 = f(1.5, 2); // double res2 = 3.5;
   * @endcode
   *
   * @note The given placeholders may actually not be present in the expression, e.g. a constant expr/function
   *        of x_ would not depends explicitly on x. Hence there is no check for this.
   *
   * @note Construction is done by CTAD, cf example above. Just pass the expression and placeholders.
   *
   * @tparam Expr Type of the expression, typically an expr<Tag, ...>, but it can be anything evaluable. 
   * @tparam PlaceholderIndex Indices of the placeholders
   */
  template <typename Expr, int... PlaceholderIndex>
  struct make_fun_impl {
    /// Expression to be evaluated.
    Expr ex;

    // internal. Use CTAD to construct
    template <typename E, auto... Is>
    make_fun_impl(E &&ex, placeholder<Is>...) : ex{std::forward<E>(ex)} {}

    // FIXME : this comment is really empty...
    /**
     * @brief Function call operator.
     *
     * @tparam Args Argument types.
     * @param args Arguments.
     * @return Result of the evaluation of the underlying expression.
     */
    template <typename... Args>
      requires(sizeof...(Args) == sizeof...(PlaceholderIndex))
    FORCEINLINE decltype(auto) operator()(Args &&...args) const {
      return eval(ex, pair<PlaceholderIndex, Args>{std::forward<Args>(args)}...);
    }
  };

  // CTAD for function
  template <typename Expr, auto... Is>
  make_fun_impl(Expr &&ex, placeholder<Is>...) -> make_fun_impl<std::decay_t<Expr>, placeholder<Is>::index...>;

  /// [deprecated] Backward compatibility maker for function. Prefer function{} instead.
  template <typename Expr, auto... Is>
  FORCEINLINE auto make_function(Expr &&ex, placeholder<Is>... p) {
    return make_fun_impl{std::forward<Expr>(ex), p...};
  }

  // is_function<T> is true iif T is a make_fun_impl
  template <typename Expr, int... Is>
  inline constexpr bool is_function<make_fun_impl<Expr, Is...>> = true;

  /** @} */

  namespace detail {

    // phset is the set of ph of the Expr, excluding the Is
    template <typename Expr, int... Is>
    constexpr uint64_t ph_set<make_fun_impl<Expr, Is...>> = (ph_set<Expr> & (~((1ull << Is) + ...)));

    // Specialization of is_lazy_impl for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    constexpr bool is_lazy_impl<make_fun_impl<Expr, Is...>> = (ph_set<make_fun_impl<Expr, Is...>> != 0);

    // // Specialization of force_copy_in_expr_impl for nda::clef::make_fun_impl types (always true).
    // template <typename Expr, int... Is>
    // constexpr bool force_copy_in_expr_impl<make_fun_impl<Expr, Is...>> = true;

  } // namespace detail

} // namespace nda::clef
