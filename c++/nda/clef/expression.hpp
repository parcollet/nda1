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
 * @brief Main expression tree class for the clef library.
 */

#pragma once

#include "./utils.hpp"
#include <type_traits>

namespace nda::clef {

  // Kind of node in the expression tree.
  enum NodeKind { Leaf, Add, Sub, Mul, Div, Eq, Leq, Geq, Less, Greater, Call, Subscript, IfElse, Loginot, UnaryPlus, Negate };

  /// Internal To dispatch the expr constructor.
  template <NodeKind K>
  inline constexpr auto node_kind = std::integral_constant<NodeKind, K>{};

  namespace detail {

    template <typename T>
    struct expr_storage : std::decay<T> {};

    template <typename T>
    struct expr_storage<T &> {
      using type = std::reference_wrapper<T>;
    };

  } // namespace detail

  /*
   * @brief Determines how a type T is stored in an expression tree.
   *
   * @details Rvalue references are moved
   *          Lvalue references are stored as a std::reference_wrapper.
   *          placeholders are an exception and always copied (cf placeholders for specialization).
   * @note INTERNAL Never be used by user directly.
   * @tparam T Type to be stored.
   */
  template <typename T>
  using expr_storage_t = typename detail::expr_storage<T>::type;

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Node of the expression tree.
   *
   * @details A recursive type, with a Kind tag and a list of child nodes.
   *
   * @note Not build by the user directly, but by combination, starting from elementary objects like placeholders.
   *       Any object can be used in the expression tree, as long as it is copyable or movable.
   * @tparam Tag   Type of the expression node (addition, function call, etc...)
   * @tparam Childs Types of the children nodes.
   */
  template <NodeKind K, typename... Childs>
  struct expr {
    static_assert(sizeof...(Childs) > 0);                  // At least one child
    static_assert(not(std::is_reference_v<Childs> | ...)); // Reference are in a reference_wrapper

    /// Children nodes of the current expression node.
    std::tuple<Childs...> childs; // FIXME in english the plural of child is ... children ?

    template <typename... Child>
    expr(std::integral_constant<NodeKind, K>, Child &&...child) : childs{std::forward<Child>(child)...} {}

    /**
     * @brief Subscript operator.
     *
     * @tparam Args Types of the subscript arguments.
     * @param args Subscript arguments.
     * @return A new node expr of Kind Subscript with children : (this, args) 
     */
#ifdef __cpp_explicit_this_parameter
    template <typename Self, typename... Args>
    auto operator[](this Self &&self, Args &&...args) {
      // NB : can not use CTAD here, as expr is the class itself...
      return expr<Subscript, expr, expr_storage_t<Args>...>{node_kind<Subscript>, std::forward<Self>(self), std::forward<Args>(args)...};
    }
#else
    // workaround for c++23 compiler without the "deducing this" implemented
    template <typename... Args>
    auto operator[](Args &&...args) const & {
      return expr<Subscript, expr, expr_storage_t<Args>...>{node_kind<Subscript>, *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator[](Args &&...args) & {
      return expr<Subscript, expr, expr_storage_t<Args>...>{node_kind<Subscript>, *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator[](Args &&...args) && {
      return expr<Subscript, expr, expr_storage_t<Args>...>{node_kind<Subscript>, std::move(*this), std::forward<Args>(args)...};
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
      return expr<Call, expr, expr_storage_t<Args>...>{node_kind<Call>, std::forward<Self>(self), std::forward<Args>(args)...};
    }
#else
    // workaround for c++23 compiler without the "deducing this" implemented
    template <typename... Args>
    auto operator()(Args &&...args) const & {
      return expr<Call, expr, expr_storage_t<Args>...>{node_kind<Call>, *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator()(Args &&...args) & {
      return expr<Call, expr, expr_storage_t<Args>...>{node_kind<Call>, *this, std::forward<Args>(args)...};
    }
    template <typename... Args>
    auto operator()(Args &&...args) && {
      return expr<Call, expr, expr_storage_t<Args>...>{node_kind<Call>, std::move(*this), std::forward<Args>(args)...};
    }
#endif
  };

  /// CTAD for expr
  //template <typename NodeTag, typename... Args>
  //expr(NodeTag, Args &&...) -> expr<NodeTag::value, expr_storage_t<Args>...>;

  template <NodeKind K, typename... Args>
  expr(std::integral_constant<NodeKind, K>, Args &&...) -> expr<K, expr_storage_t<Args>...>;

  namespace detail {

    // ph_set of an expr is the union of the ph_set of the children
    template <NodeKind K, typename... Ts>
    constexpr uint64_t ph_set<expr<K, Ts...>> = (ph_set<Ts> | ...);

    // An expr is lazy.
    template <NodeKind K, typename... Ts>
    constexpr bool is_lazy_impl<expr<K, Ts...>> = true;

  } // namespace detail

  /** @} */

} // namespace nda::clef
