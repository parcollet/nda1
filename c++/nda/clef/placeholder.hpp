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
#include <cstdint>
#include <utility>

#include "../macros.hpp"
#include "./expression.hpp"
#include "./utils.hpp"

namespace nda::clef {

  /**
   * @addtogroup clef_placeholders
   * @{
   */

  /**
   * @brief A pair consisting of a placeholder index and its assigned value.
   *
   * @details Only created with the syntax placeholder = x. 
   *
   * @tparam N Placeholder index.
   * @tparam T Value type.
   */

  template <int N, typename T>
  struct ph_value_pair {
    /// Value assigned to the placeholder (can be an lvalue reference).
    T value;

    /// Integer label of the placeholder.
    static constexpr int idx = N;
  };

  //-------------------------------------------------------------------

  /**
   * @brief A placeholder for lazy expressions.
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto expr = i_ + j_;
   * auto res = eval(expr, i_ = 1.0, j_ = 2.0); // ---> double res = 3.0;
   * @endcode
   *
   * @tparam N Index (must be < 64).
   */
  template <int N>
  struct placeholder {
    // We rely on it as we use 64bits uint in some operations at compile time.
    // but it could be generalized.
    static_assert(N >= 0 && N < 64, "Placeholder index must be in {0, 1, ..., 63}");

    /// Index of the placeholder
    static constexpr int index = N;

    /**
     *
     *
     * @tparam T Type of the value
     * @param x The value
     * @return A ph_value_pair object containing x
     */
    template <typename T>
    FORCEINLINE ph_value_pair<N, T> operator=(T &&x) const { // NOLINT (we want to return a pair)
      return {std::forward<T>(x)};
    }

    /**
     * @brief Function call operator. 
     *
     * @tparam Args Arguments types.
     * @param args Arguments
     * @return A node (of type expr) represending this(args)
     */
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return expr{node_kind<Call>, auto{*this}, std::forward<Args>(args)...}; // auto{} : we copy the ph anyway
    }

    /**
     * @brief Subscript operator.
     *
     * @tparam Args Arguments types.
     * @param args Subscript arguments
     * @return A node (of type expr) represending this[args]
     */
    template <typename... Args>
    auto operator[](Args &&...args) const {
      return expr{node_kind<Subscript>, auto{*this}, std::forward<Args>(args)...};
    }
  };

  // ------------------------------

  namespace detail {
    // placeholder are always copied. They are empty anyway, but it greatly
    // simplify pattern recognition in the auto_assign later.
    // We specialize expr_storage for this type.
    template <int N>
    struct expr_storage<placeholder<N> &> {
      using type = placeholder<N>;
    };

    template <int N>
    struct expr_storage<placeholder<N> const &> {
      using type = placeholder<N>;
    };

    // For a placeholder, the ph_set contains just N.
    template <int N>
    constexpr uint64_t ph_set<placeholder<N>> = 1ull << N;

    // placeholder are lazy objects.
    template <int N>
    constexpr bool is_lazy_impl<placeholder<N>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef
