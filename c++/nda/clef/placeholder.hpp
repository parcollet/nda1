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
 * @brief Provides placeholders for the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"

#include <cstdint>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_placeholders
   * @{
   */

  /**
   * @brief A pair consisting of a placeholder index and its assigned value.
   *
   * @details The user does not explicitly create or handle pair objects. 
   *          Use placeholder = x syntax, cf placeholder.
   *          (see nda::clef::placeholder for an example).
   *
   * @tparam N Placeholder index.
   * @tparam T Value type.
   */
  template <int N, typename T>
  struct pair {
    /// Value assigned to the placeholder (can be an lvalue reference).
    T rhs;

    /// Integer label of the placeholder.
    static constexpr int p = N;
  };

  /**
   * @brief A placeholder. It is an empty struct, labelled by an index (int).
   *
   * @details It is the basic building block of lazy expressions. For example:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto expr = i_ + j_;
   * auto res = nda::clef::eval(expr, i_ = 1.0, j_ = 2.0); // double res = 3.0;
   * @endcode
   *
   * Here `expr` is a lazy binary nda::clef::expr with the nda::clef::tags::plus tag, which can be evaluated later on
   * with the nda::clef::eval function and by assigning values to the placeholders (see nda::clef::pair).
   *
   * @tparam N Index (must be < 64).
   */
  template <int N>
  struct placeholder {
    static_assert(N >= 0 && N < 64, "Placeholder index must be in {0, 1, ..., 63}");

    /// Index
    static constexpr int index = N;

    /**
     * @brief Build a (placeholder, value) pair
     *
     * @tparam RHS Type of the right-hand side.
     * @param rhs Right-hand side of the assignment.
     * @return An nda::clef::pair object. It basically tags the value with the placeholder index.
     */
    template <typename RHS>
    pair<N, RHS> operator=(RHS &&rhs) const { // NOLINT (we want to return a pair)
      return {std::forward<RHS>(rhs)};
    }

    /**
     * @brief Function call operator. 
     *
     * @tparam Args Arguments types.
     * @param args Arguments
     * @return An expression node (nda::clef::expr) represending a function call of this with the given arguments.
     */
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return expr{tags::function{}, auto{*this}, std::forward<Args>(args)...}; // auto{} we copy the ph anyway
    }

    /**
     * @brief Subscript operator.
     *
     * @tparam Args Arguments types.
     * @param args Subscript arguments
     * @return An expression node (nda::clef::expr) represending a [] call of this with the given arguments.
     */
    template <typename... Args>
    auto operator[](Args &&...args) const {
      return expr{tags::subscript{}, auto{*this}, std::forward<Args>(args)...};
    }
  };

  // ------------------------------

  namespace detail {
    // placeholder are always copied. They are empty anyway, but it greatly
    // simplify pattern recognition in the auto_assign
    // We specialize expr_storage for this type.
    template <int N>
    struct expr_storage_impl<placeholder<N> &> {
      using type = placeholder<N>;
    };

    template <int N>
    struct expr_storage_impl<placeholder<N> const &> {
      using type = placeholder<N>;
    };

    // Specialization of ph_set for nda::clef::placeholder types.
    template <int N>
    constexpr uint64_t ph_set<placeholder<N>> = 1ull << N;

    // Specialization of is_lazy_impl for nda::clef::placeholder types.
    template <int N>
    constexpr bool is_lazy_impl<placeholder<N>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef
