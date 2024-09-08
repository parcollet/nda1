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

#include "./utils.hpp"
#include "./placeholder.hpp"
#include "../macros.hpp"

#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Helper struct to simplify calls to nda::clef::eval.
   *
   * @details It stores the object (usually a lazy expression) which we want to evaluate and takes integer labels of
   * placeholders as template arguments.
   *
   * See nda::clef::make_function for an example.
   *
   * @tparam T Type of the object.
   * @tparam Is Integer labels of the placeholders in the expression.
   */
  template <typename Expr, int... Is>
  struct make_fun_impl {
    /// Expression to be evaluated.
    Expr ex;

    /**
     * @brief Function call operator.
     *
     * @details The arguments together with the integer labels (template parameters of the class) are used to construct
     * nda::clef::pair objects. The stored object and the pairs are then passed to the nda::clef::eval function to
     * perform the evaluation.
     *
     * @tparam Args Types of the function call arguments.
     * @param args Function call arguments.
     * @return Result of the evaluation.
     */
    template <typename... Args>
    FORCEINLINE decltype(auto) operator()(Args &&...args) const {
      return eval(ex, pair<Is, Args>{std::forward<Args>(args)}...);
    }
  };

  /**
   * @brief Factory function for nda::clef::make_fun_impl objects.
   *
   * @details The given arguments are used to construct a new nda::clef::make_fun_impl object. The first argument is
   * forwarded to its constructor and the integer labels of the remaining placeholder arguments are used in its template
   * argument list.
   *
   * The following example shows how to turn a binary lazy expression `ex` into a callable object `f` that takes two
   * arguments:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto f = nda::clef::make_function(ex, i_, j_);
   * auto res = f(1, 2);    // int res = 3;
   * auto res2 = f(1.5, 2); // double res2 = 3.5;
   * @endcode
   *
   * @note There is no check if the given placeholders actually match placeholders in the object to be evaluated.
   *
   * @tparam T Type of the object.
   * @tparam Phs Types of the placeholders.
   * @param obj Object to be stored in the nda::clef::make_fun_impl object.
   * @return A callable nda::clef::make_fun_impl object that takes as many arguments as placeholders were given.
   */
  template <typename T, typename... Phs>
  FORCEINLINE auto make_function(T &&obj, Phs...) {
    return make_fun_impl<std::decay_t<T>, Phs::index...>{std::forward<T>(obj)};
  }

  /** @} */

  namespace detail {

    // Specialization of is_function_impl for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    inline constexpr bool is_function_impl<make_fun_impl<Expr, Is...>> = true;

    // phset is the set of ph of the Expr, excluding the Is
    template <typename Expr, int... Is>
    constexpr uint64_t ph_set<make_fun_impl<Expr, Is...>> = (ph_set<Expr> & (~((1ull << Is) + ...)));

    // Specialization of is_lazy_impl for nda::clef::make_fun_impl types.
    template <typename Expr, int... Is>
    constexpr bool is_lazy_impl<make_fun_impl<Expr, Is...>> = (ph_set<make_fun_impl<Expr, Is...>> != 0);

    // Specialization of force_copy_in_expr_impl for nda::clef::make_fun_impl types (always true).
    template <typename Expr, int... Is>
    constexpr bool force_copy_in_expr_impl<make_fun_impl<Expr, Is...>> = true;

  } // namespace detail

} // namespace nda::clef
