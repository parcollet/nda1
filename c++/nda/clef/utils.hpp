
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
 * @brief Provides some utility functions and type traits for the CLEF library.
 */

#pragma once

#include <functional>
#include <type_traits>

namespace nda::clef {

  /**
   * @addtogroup clef_utils
   * @{
   */

  namespace detail {

    // Helper variable to determine if an objects of type `T` should be forced to be copied into an expression tree.
    template <typename T>
    constexpr bool force_copy_in_expr_impl = false;

    // Helper struct to determine how a type should be stored in an expression tree.
    template <class T>
    struct expr_storage_impl {
      using type = T;
    };

    // FIXME :
    // default : decay_t<T>
    // T & -> reference_wrapper
    // remove force_copy_in_expr_impl
    // replace by a specialization of expr_storage_impl<T>

    // WHY DO WE STILL NEED THIS ????

    // Specialization of expr_storage_impl for lvalue references.
    template <class T>
    struct expr_storage_impl<T &>
       : std::conditional<force_copy_in_expr_impl<std::remove_const_t<T>>, std::remove_const_t<T>, std::reference_wrapper<T>> {};

    // Specialization of expr_storage_impl for rvalue references.
    template <class T>
    struct expr_storage_impl<T &&> {
      using type = T;
    };

    // Specialization of expr_storage_impl for const rvalue references.
    template <class T>
    struct expr_storage_impl<const T &&> {
      using type = T;
    };

    // Specialization of expr_storage_impl for const types.
    template <class T>
    struct expr_storage_impl<const T> {
      using type = T;
    };

    // Type used to encode which placeholder indices are used in an expression.
    using ull_t = unsigned long long;

    template <typename T>
    constexpr uint64_t ph_set = 0;

    // Helper variable to determine if a type `T` is lazy.
    template <typename T>
    constexpr bool is_lazy_impl = false;

    // Specialization of is_lazy_impl for cvref types.
    template <typename T>
      requires(!std::is_same_v<T, std::remove_cvref_t<T>>)
    constexpr bool is_lazy_impl<T> = is_lazy_impl<std::remove_cvref_t<T>>;

    // Helper variable to determine if a type `T` is an make_fun_impl type.
    template <typename T>
    inline constexpr bool is_function_impl = false;

    // An erroneous diagnostics in gcc: i0 is indeed used. We silence it.
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

    // Check if all given integers are different.
    template <typename... Is>
    constexpr bool all_different(int i0, Is... is) {
      return (((is - i0) * ... * 1) != 0);
    }

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif

  } // namespace detail

  /// Constexpr variable that is true if objects of type `T` should be forced to be copied into an expression tree.
  template <typename T>
  constexpr bool force_copy_in_expr = false; //detail::force_copy_in_expr_impl<T>;

  /**
   * @brief Type trait to determine how a type should be stored in an expression tree, i.e. either by reference or by
   * value?
   *
   * @details Rvalue references are copied/moved into the expression tree. Lvalue references are stored as a
   * std::reference_wrapper, unless the compile-time value of nda::clef::force_copy_in_expr is true.
   *
   * @tparam T Type to be stored.
   */
  template <class T>
  using expr_storage_t = typename detail::expr_storage_impl<T>::type;

  // FIXME : why not specialize is_lazy directly ???

  /// true iif T is a lazy type.
  template <typename T>
  constexpr bool is_lazy = detail::is_lazy_impl<T>;

  /// true iff at least one of the Ts is lazy
  template <typename... Ts>
  constexpr bool is_any_lazy = (is_lazy<Ts> or ...);

  // FIXME : remove ?
  /// Alias template for nda::clef::is_any_lazy.
  template <typename... Ts>
  constexpr bool is_clef_expression = is_any_lazy<Ts...>;

  /// MOVE THIS
  /// Constexpr variable that is true if the type `T` is an nda::clef::make_fun_impl type.
  template <typename T>
  inline constexpr bool is_function = detail::is_function_impl<T>;

  /** @} */

} // namespace nda::clef
