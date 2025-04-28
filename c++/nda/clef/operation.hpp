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
#include <utility>

#include "./expression.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

namespace nda::clef {
  namespace detail {

    // Generic operation like std::plus<void>
    // We need more than is in the std, and we want to enforce always_inline
    template <NodeKind K>
    struct operation;

    template <>
    struct operation<NodeKind::Call> {
      template <typename F, typename... Args>
      FORCEINLINE static decltype(auto) invoke(F &&f, Args &&...args) {
        // If the F does not have an lazy able (), we make the expression
#if 1
        if constexpr ((is_lazy<Args> or ...))
          return expr{node_kind<Call>, std::forward<F>(f), std::forward<Args>(args)...};
        else
          return std::forward<F>(f)(std::forward<Args>(args)...);
#else
        // if constexpr (requires { std::forward<F>(f)(std::forward<Args>(args)...); })
        return std::forward<F>(f)(std::forward<Args>(args)...);
        // else
        //  return expr{tags::function{}, std::forward<F>(f), std::forward<Args>(args)...};
#endif
      }
    };

    // [] operator, similar to function
    template <>
    struct operation<NodeKind::Subscript> {
      template <typename F, typename... Args>
      FORCEINLINE static decltype(auto) invoke(F &&f, Args &&...args) {
        // If the F does not have an lazy able [], we make the expression
        if constexpr ((is_lazy<Args> or ...))
          return expr{node_kind<Subscript>, std::forward<F>(f), std::forward<Args>(args)...};
        else {
          // We call the [] operator.
          // BUT we add a protection. In the case where F is an rvalue ref,
          // and f[...] returns a reference, it is highly suspicious,
          // like a vector v[0] would return a dangling reference.
          // So we return a COPY of the value, not a reference.
          // FIXME : this behavious could be overriden by a trait in the future
          if constexpr (std::is_rvalue_reference_v<F &&> && std::is_reference_v<decltype(std::forward<F>(f)[std::forward<Args>(args)...])>) {
            return auto{std::forward<F>(f)[std::forward<Args>(args)...]};
          } else {
            return std::forward<F>(f)[std::forward<Args>(args)...];
          }
        }
      }
    };
  } // namespace detail

  // ------------------------ arithmetic operations --------------------------

// Define and implement all lazy binary operations.
#define DEFINE_CLEF_OPERATION(TAG, OP)                                                                                                               \
  /** @brief The `OP` operation for lazy object. Returns a clef::expr */                                                                             \
  template <typename L, typename R>                                                                                                                  \
    requires(is_lazy<L> or is_lazy<R>)                                                                                                               \
  FORCEINLINE auto operator OP(L &&l, R &&r) {                                                                                                       \
    return expr{node_kind<TAG>, std::forward<L>(l), std::forward<R>(r)};                                                                             \
  }

  // clang-format off
  DEFINE_CLEF_OPERATION(NodeKind::Add, +);
  DEFINE_CLEF_OPERATION(NodeKind::Sub, -);
  DEFINE_CLEF_OPERATION(NodeKind::Mul, *);
  DEFINE_CLEF_OPERATION(NodeKind::Div, /);
  DEFINE_CLEF_OPERATION(NodeKind::Greater, >);
  DEFINE_CLEF_OPERATION(NodeKind::Less, <);
  DEFINE_CLEF_OPERATION(NodeKind::Leq, <=);
  DEFINE_CLEF_OPERATION(NodeKind::Geq, >=);
  DEFINE_CLEF_OPERATION(NodeKind::Eq, ==);
// clang-format on
#undef DEFINE_CLEF_OPERATION

// Define and implement all lazy unary operations.
#define DEFINE_CLEF_OPERATION(TAG, OP)                                                                                                               \
  /** @brief Implementation of the lazy unary `OP` operation. */                                                                                     \
  template <typename L>                                                                                                                              \
    requires(is_lazy<L>)                                                                                                                             \
  FORCEINLINE auto operator OP(L &&l) {                                                                                                              \
    return expr{node_kind<TAG>, std::forward<L>(l)};                                                                                                 \
  }

  DEFINE_CLEF_OPERATION(UnaryPlus, +);
  DEFINE_CLEF_OPERATION(Negate, -);
  DEFINE_CLEF_OPERATION(Loginot, !);
#undef DEFINE_CLEF_OPERATION

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
    return expr{node_kind<IfElse>, std::forward<C>(c), std::forward<A>(a), std::forward<B>(b)};
  }

} // namespace nda::clef
