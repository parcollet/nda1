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
 * @brief Provides operations for the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Generic operation performed on expression nodes.
   *
   * @details Specializations for the different operation tags provide an implementation of the function call operator
   * which performs the actual operation on the given operands.
   *
   * @tparam Tag Tag of the operation.
   */
  template <typename Tag>
  struct operation;

  /// Specialization of nda::clef::operation for nda::clef::tags::terminal.
  template <>
  struct operation<tags::terminal> {
    /**
     * @brief Perform a terminal operation.
     *
     * @tparam L Type of the operand.
     * @param l Operand.
     * @return Forwarded lvalue/rvalue reference of the argument.
     */
    template <typename L>
    FORCEINLINE static L invoke(L &&l) {
      return std::forward<L>(l);
    }
  };

  /// Specialization of nda::clef::operation for nda::clef::tags::function.
  template <>
  struct operation<tags::function> {
    /**
     * @brief Perform a function call operation.
     *
     * @tparam F Type of the callable.
     * @tparam Args Types of the function call arguments.
     * @param f Callable object.
     * @param args Function call arguments.
     * @return Result of the function call.
     */
    template <typename F, typename... Args>
    FORCEINLINE static decltype(auto) invoke(F &&f, Args &&...args) {
      return std::forward<F>(f)(std::forward<Args>(args)...);
    }
  };

  /// Specialization of nda::clef::operation for nda::clef::tags::subscript.
  template <>
  struct operation<tags::subscript> {
    /**
     * @brief Perform a subscript operation.
     *
     * @tparam F Type of the object to be subscripted.
     * @tparam Args Types of the subscript arguments.
     * @param f Object to be subscripted.
     * @param args Subscript arguments.
     * @return Result of the subscript operation.
     */
    template <typename F, typename... Args>
    FORCEINLINE static decltype(auto) invoke(F &&f, Args &&...args) {
      // directly calling [args...] breaks clang
      return std::forward<F>(f).operator[](std::forward<Args>(args)...);
    }
  };

  // Define and implement all lazy binary operations.
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    /** @brief Tag for binary `OP` expressions. */                                                                                                   \
    struct TAG : binary_op {                                                                                                                         \
      /** @brief String representation of the operation. */                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  /** @brief Implementation of the lazy binary `OP` operation. */                                                                                    \
  template <typename L, typename R>                                                                                                                  \
  FORCEINLINE auto operator OP(L &&l, R &&r)                                                                                                         \
    requires(is_lazy<L> or is_lazy<R>)                                                                                                               \
  {                                                                                                                                                  \
    return expr{tags::TAG{}, std::forward<L>(l), std::forward<R>(r)};                                                                                \
  }                                                                                                                                                  \
  /** @brief Specialization of nda::clef::operation for nda::clef::tags::TAG. */                                                                     \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    /** @brief Function call operator to perform the actual binary `OP` operation. */                                                                \
    template <typename L, typename R>                                                                                                                \
    FORCEINLINE static decltype(auto) invoke(L &&l, R &&r) {                                                                                         \
      return std::forward<L>(l) OP std::forward<R>(r);                                                                                               \
    }                                                                                                                                                \
  }

  // clang-format off
  CLEF_OPERATION(plus, +);
  CLEF_OPERATION(minus, -);
  CLEF_OPERATION(multiplies, *);
  CLEF_OPERATION(divides, /);
  CLEF_OPERATION(greater, >);
  CLEF_OPERATION(less, <);
  CLEF_OPERATION(leq, <=);
  CLEF_OPERATION(geq, >=);
  CLEF_OPERATION(eq, ==);
  // clang-format on
#undef CLEF_OPERATION

  // Define and implement all lazy unary operations.
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    /** @brief Tag for unary `OP` expressions. */                                                                                                    \
    struct TAG : unary_op {                                                                                                                          \
      /** @brief String representation of the operation. */                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  /** @brief Implementation of the lazy unary `OP` operation. */                                                                                     \
  template <typename L>                                                                                                                              \
  FORCEINLINE auto operator OP(L &&l)                                                                                                                \
    requires(is_any_lazy<L>)                                                                                                                         \
  {                                                                                                                                                  \
    return expr{tags::TAG{}, std::forward<L>(l)};                                                                                                    \
  }                                                                                                                                                  \
  /** @brief Specialization of nda::clef::operation for nda::clef::tags::TAG. */                                                                     \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    /** @brief Function call operator to perform the actual unary `OP` operation. */                                                                 \
    template <typename L>                                                                                                                            \
    FORCEINLINE static decltype(auto) invoke(L &&l) {                                                                                                \
      return OP std::forward<L>(l);                                                                                                                  \
    }                                                                                                                                                \
  }

  CLEF_OPERATION(unaryplus, +);
  CLEF_OPERATION(negate, -);
  CLEF_OPERATION(loginot, !);
#undef CLEF_OPERATION

  /// Specialization of nda::clef::operation for nda::clef::tags::if_else.
  template <>
  struct operation<tags::if_else> {
    /**
     * @brief Perform a ternary (if-else) operation.
     *
     * @tparam C Type of the condition.
     * @tparam A Type of the return type when the condition is true.
     * @tparam B Type of the return type when the condition is false.
     * @param c Condition convertible to bool.
     * @param a Return value when the condition is true.
     * @param b Return value when the condition is false (needs to be convertible to A).
     * @return Result of the ternary operation.
     */
    template <typename C, typename A, typename B>
    FORCEINLINE static A invoke(C const &c, A &&a, B &&b) {
      return c ? std::forward<A>(a) : std::forward<B>(b);
    }
  };

  /**
   * @brief Create a lazy ternary (if-else) expression.
   *
   * @tparam C Type of the conditional expression.
   * @tparam A Type of the return expression when the condition is true.
   * @tparam B Type of the return expression when the condition is false.
   * @param c Conditional expression.
   * @param a Return expression when the condition is true.
   * @param b Return expression when the condition is false.
   * @return An nda::clef::expr object with the nda::clef::tags::ternary tag and the given
   * operands forwarded as its child nodes.
   */
  template <typename C, typename A, typename B>
  FORCEINLINE auto if_else(C &&c, A &&a, B &&b) {
    return expr{tags::if_else(), std::forward<C>(c), std::forward<A>(a), std::forward<B>(b)};
  }
  /** @} */

} // namespace nda::clef
