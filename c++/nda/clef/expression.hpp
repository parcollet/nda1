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
 * @brief Provides a basic lazy expression type for the clef library.
 */

#pragma once

#include "./utils.hpp"

#include <tuple>
#include <utility>

namespace nda::clef {

  namespace tags {

    /**
     * @addtogroup clef_expr
     * @{
     */

    /// Tag for function call expressions.
    struct function {};

    /// Tag for subscript expressions.
    struct subscript {};

    /// Tag to indicate a terminal node in the expression tree.
    struct terminal {};

    /// Tag for conditional expressions.
    struct if_else {};

    /// Tag for unary operator expressions.
    struct unary_op {};

    /// Tag for binary operator expressions.
    struct binary_op {};

    /** @} */

  } // namespace tags

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Single node of the expression tree.
   *
   * @details An expression node contains a tag that determines the type of expression and a tuple of child nodes which
   * are usually either other expression nodes, nda::clef::placeholder objects or specific values/objects like an int or
   * a double.
   *
   * @tparam Tag Tag of the expression.
   * @tparam Ts Types of the child nodes.
   */
  template <typename Tag, typename... Ts>
  struct expr {
    /// Tuple type for storing the child nodes.
    using childs_t = std::tuple<Ts...>;

    /// Child nodes of the current expression node.
    childs_t childs;

    /// Default copy constructor.
    expr(expr const &) = default;

    /**
     * @brief Move constructor simply moves the child nodes from the source expression.
     * @param ex Source expression.
     */
    expr(expr &&ex) noexcept : childs(std::move(ex.childs)) {}

    /// Copy assignment operator is deleted.
    expr &operator=(expr const &) = delete;

    /// Default move assignment operator.
    expr &operator=(expr &&) = default;

    /**
     * @brief Construct an expression node with a given tag and child nodes.
     *
     * @tparam Us Types of the child nodes.
     * @param us Child nodes.
     */
    template <typename... Us>
    expr(Tag, Us &&...us) : childs(std::forward<Us>(us)...) {}

    /**
     * @brief Subscript operator.
     *
     * @tparam Args Types of the subscript arguments.
     * @param args Subscript arguments.
     * @return An nda::clef::expr object with the nda::clef::tags::subscript tag containing the current expression node
     * as the first child node and the other arguments as the remaining child nodes.
     */
    template <typename... Args>
    auto operator[](Args &&...args) const {
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript(), *this, std::forward<Args>(args)...};
    }

    /**
     * @brief Function call operator.
     *
     * @tparam Args Types of the function call arguments.
     * @param args Function call arguments.
     * @return An nda::clef::expr object with the nda::clef::tags::function tag containing the current expression node
     * as the first child node and the other arguments as the remaining child nodes.
     */
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function(), *this, std::forward<Args>(args)...};
    }
  };

  /// CTAD for expr
  template <typename Tag, typename... Args>
  expr(Tag, Args &&...) -> expr<Tag, expr_storage_t<Args>...>;

  namespace detail {

    // Specialization of ph_set for nda::clef::expr types.
    template <typename Tag, typename... Ts>
    struct ph_set<expr<Tag, Ts...>> : ph_set<Ts...> {};

    // Specialization of is_lazy_impl for nda::clef::expr types (always true).
    template <typename Tag, typename... Ts>
    constexpr bool is_lazy_impl<expr<Tag, Ts...>> = true;

    // Specialization of force_copy_in_expr_impl for nda::clef::expr types (always true).
    template <typename Tag, typename... Ts>
    constexpr bool force_copy_in_expr_impl<expr<Tag, Ts...>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef
