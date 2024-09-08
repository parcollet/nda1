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
#include <cstdint>
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
  namespace detail {

    // Helper struct to determine how a type should be stored in an expression tree.
    template <typename T>
    struct expr_storage_impl : std::decay<T> {};

    // Specialization of expr_storage_impl for lvalue references.
    template <typename T>
    struct expr_storage_impl<T &> {
      using type = std::reference_wrapper<T>;
    };
    // NB : placeholder will also be specialized to make a copy, cf placeholder.

  } // namespace detail

  /**
   * @brief Trait to determine how a type should be stored in an expression tree, i.e. either by reference or by value?
   *
   * @details Rvalue references are copied/moved into the expression tree.
   *          Lvalue references are stored as a std::reference_wrapper.
   *          placeholders are an exception and always copied (cf placeholders for specialization).
   * @note Should never be used by user directly.
   * @tparam T Type to be stored.
   */
  template <typename T>
  using expr_storage_t = typename detail::expr_storage_impl<T>::type;

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Node of the expression tree.
   *
   * @details An expression node contains a tag that determines the type of expression and a tuple of child nodes which
   * are usually either other expression nodes, nda::clef::placeholder objects or other objects (e.g. int, double)
   *
   * @note expr are not build by the user directly, but by combination, starting from placeholder.
   * @tparam Tag   Type of the expression node (addition, function call, etc...)
   * @tparam Childs Types of the children nodes.
   */
  template <typename Tag, typename... Childs>
  struct expr {

    /// Children nodes of the current expression node.
    std::tuple<Childs...> childs; // FIXME in english the plural of child is ... children ?

    expr(expr const &)            = default;
    expr(expr &&)                 = default;
    expr &operator=(expr const &) = delete;
    expr &operator=(expr &&)      = default;

    /// Construct from the tag and children nodes. Tag is useful here (for CTAD e.g.)
    template <typename... Child>
    expr(Tag, Child &&...child) : childs{std::forward<Child>(child)...} {}

    /**
     * @brief Subscript operator.
     *
     * @tparam Args Types of the subscript arguments.
     * @param args Subscript arguments.
     * @return An nda::clef::expr object with the nda::clef::tags::subscript tag containing the current expression node
     * as the first child node and the other arguments as the remaining child nodes.
     */
#ifdef __cpp_explicit_this_parameter
    template <typename Self, typename... Args>
    auto operator[](this Self &&self, Args &&...args) {
      // NB : can not use CTAD here, as expr is the class itself...
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript{}, std::forward<Self>(self), std::forward<Args>(args)...};
    }
#else
    // workaround for c++23 compiler without the "deducing this" implemented
    template <typename... Args>
    auto operator[](Args &&...args) const & {
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript(), *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator[](Args &&...args) & {
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript(), *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator[](Args &&...args) && {
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript(), std::move(*this), std::forward<Args>(args)...};
    }
#endif

/**
     * @brief Function call operator.
     *
     * @tparam Args Types of the function call arguments.
     * @param args Function call arguments.
     * @return An nda::clef::expr object with the nda::clef::tags::function tag containing the current expression node
     * as the first child node and the other arguments as the remaining child nodes.
     */
#ifdef __cpp_explicit_this_parameter
    template <typename Self, typename... Args>
    auto operator()(this Self &&self, Args &&...args) {
      // NB : can not use CTAD here, as expr is the class itself...
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function{}, std::forward<Self>(self), std::forward<Args>(args)...};
    }
#else
    template <typename... Args>
    auto operator()(Args &&...args) const & {
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function(), *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator()(Args &&...args) & {
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function(), *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator()(Args &&...args) && {
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function(), std::move(*this), std::forward<Args>(args)...};
    }
#endif
  };

  /// CTAD for expr
  template <typename Tag, typename... Args>
  expr(Tag, Args &&...) -> expr<Tag, expr_storage_t<Args>...>;

  namespace detail {

    // ph_set of an expr is the union of the ph_set of the children
    template <typename Tag, typename... Ts>
    constexpr uint64_t ph_set<expr<Tag, Ts...>> = (ph_set<Ts> | ...);

    // Specialization of is_lazy_impl for nda::clef::expr types (always true).
    template <typename Tag, typename... Ts>
    constexpr bool is_lazy_impl<expr<Tag, Ts...>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef
