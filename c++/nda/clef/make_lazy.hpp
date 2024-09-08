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
 * @brief Provides functionality to make objects, functions and methods lazy.
 */

#pragma once

#include "./expression.hpp"
#include "./utils.hpp"

#include <utility>

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  /**
   * @brief Create a terminal expression node of an object.
   *
   * @tparam T Type of the object.
   * @param t Given object.
   * @return An nda::clef::expr with the nda::clef::tags::terminal tag containing either a copy of the object itself
   * (if it is moved) or a reference to the object.
   */
  template <typename T>
  auto make_expr(T &&t) {
    return expr{tags::terminal(), std::forward<T>(t)};
  }

  // FIXME : DEPRECATE : it is now just make_expr(auto{something});
  /**
   * @brief Create a terminal expression node of an object.
   *
   * @tparam T Type of the object.
   * @param t Given object.
   * @return An nda::clef::expr with the nda::clef::tags::terminal tag containing a copy of the object.
   */
  template <typename T>
  auto make_expr_from_clone(T &&x) {
    return make_expr(auto{x});
  }

  /**
   * @brief Create a function call expression from a callable object and a list of arguments.
   *
   * @tparam F Type of the callable object.
   * @tparam Args Types of the arguments.
   * @param f Callable object.
   * @param args Function arguments.
   * @return An nda::clef::expr with the nda::clef::tags::function tag containing the callable and the forwarded
   * arguments.
   */
  template <typename F, typename... Args>
  auto make_expr_call(F &&f, Args &&...args)
    requires((is_lazy<Args> || ...))
  {
    return expr{tags::function{}, std::forward<F>(f), std::forward<Args>(args)...};
  }

  /**
   * @brief Create a subscript expression from an object and a list of arguments.
   *
   * @tparam T Type of the object to be subscripted.
   * @tparam Args Types of the arguments.
   * @param t Object to be subscripted.
   * @param args Subscript arguments.
   * @return An nda::clef::expr with the nda::clef::tags::subscript tag containing the subscriptable object and the
   * forwarded arguments.
   */
  template <typename T, typename... Args>
  auto make_expr_subscript(T &&t, Args &&...args)
    requires((is_lazy<Args> || ...))
  {
    return expr{tags::subscript{}, std::forward<T>(t), std::forward<Args>(args)...};
  }

  /// Macro to make any function lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_MAKE_FNT_LAZY(name)                                                                                                                     \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a)                                                                                                                              \
    requires(nda::clef::is_any_lazy<A...>)                                                                                                           \
  {                                                                                                                                                  \
    return make_expr_call([](auto &&...__b) -> decltype(auto) { return name(std::forward<decltype(__b)>(__b)...); }, std::forward<A>(__a)...);       \
  }

  /// Macro to make any method lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_IMPLEMENT_LAZY_METHOD(TY, name)                                                                                                         \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a)                                                                                                                              \
    requires(nda::clef::is_any_lazy<A...>)                                                                                                           \
  {                                                                                                                                                  \
    return make_expr_call(                                                                                                                           \
       [](auto &&__obj, auto &&...__b) -> decltype(auto) { return std::forward<decltype(__obj)>(__obj).name(std::forward<decltype(__b)>(__b)...); }, \
       *this, std::forward<A>(__a)...);                                                                                                              \
  }

  /// Macro to make any function call operator lazy, i.e. accept lazy arguments and return a function call expression node.
#define CLEF_IMPLEMENT_LAZY_CALL(...)                                                                                                                \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) const &                                                                                                            \
    requires(nda::clef::is_any_lazy<Args...>)                                                                                                        \
  {                                                                                                                                                  \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) &                                                                                                                  \
    requires(nda::clef::is_any_lazy<Args...>)                                                                                                        \
  {                                                                                                                                                  \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) &&                                                                                                                 \
    requires(nda::clef::is_any_lazy<Args...>)                                                                                                        \
  {                                                                                                                                                  \
    return make_expr_call(std::move(*this), std::forward<Args>(args)...);                                                                            \
  }

  /** @} */

} // namespace nda::clef
