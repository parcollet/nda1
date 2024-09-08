// Copyright (c) 2019-2021 Simons Foundation
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
 * @brief Provides IO for clef objects.
 */

#pragma once

#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "./operation.hpp"
#include "./placeholder.hpp"
#include "./expression.hpp"
#include "./function.hpp"

namespace nda::clef {

  /**
   * @addtogroup clef_utils
   * @{
   */

  /**
   * @brief Print an nda::clef::placeholder to a std::ostream.
   *
   * @tparam N Integer label of the placeholder.
   * @param sout std::ostream object to print to.
   * @return Reference to the std::ostream.
   */
  template <int N>
  std::ostream &operator<<(std::ostream &sout, placeholder<N>) {
    return sout << "_" << N;
  }

  /**
   * @brief Print the value contained in a std::reference_wrapper to std::ostream.
   *
   * @tparam T Value type of the std::reference_wrapper.
   * @param sout std::ostream object to print to.
   * @param wrapper std::reference_wrapper object to print.
   * @return Reference to the std::ostream.
   */
  template <typename T>
  std::ostream &operator<<(std::ostream &sout, std::reference_wrapper<T> const &wrapper) {
    return sout << wrapper.get();
  }

  // Overload of nda::clef::variadic_print for the case of an empty list of arguments.
  inline std::ostream &variadic_print(std::ostream &sout) { return sout; }

  /**
   * @brief Print a variadic list of arguments to std::ostream.
   *
   * @tparam T0 Type of the first argument.
   * @tparam Ts Types of the remaining arguments.
   * @param sout std::ostream object to print to.
   * @param t0 First argument to print.
   * @param ts Remaining arguments to print.
   * @return Reference to the std::ostream.
   */
  template <typename T0, typename... Ts>
  std::ostream &variadic_print(std::ostream &sout, T0 &&t0, Ts &&...ts) {
    sout << std::forward<T0>(t0) << (sizeof...(Ts) > 0 ? ", " : "");
    variadic_print(sout, std::forward<Ts>(ts)...);
    return sout;
  }

  /**
   * @brief Print a std::tuple to std::ostream.
   *
   * @tparam Tuple Type of the std::tuple to print.
   * @param sout std::ostream object to print to.
   * @param t std::tuple object to print.
   * @return Reference to the std::ostream.
   */
  template <typename Tuple>
  std::ostream &print_tuple(std::ostream &sout, Tuple const &t) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (void(sout << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    return sout;
  }

  /**
   * @brief Print an nda::clef::tags::unary_op expression to std::ostream.
   *
   * @tparam Tag Type of the unary operation.
   * @tparam L Type of the child expression.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename Tag, typename L>
    requires std::is_base_of_v<tags::unary_op, Tag>
  std::ostream &operator<<(std::ostream &sout, expr<Tag, L> const &ex) {
    return sout << "(" << Tag::name() << " " << std::get<0>(ex.childs) << ")";
  }

  /**
   * @brief Print an nda::clef::tags::binary_op expression to std::ostream.
   *
   * @tparam Tag Type of the binary operation.
   * @tparam L Type of the child expression #1.
   * @tparam R Type of the child expression #2.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename Tag, typename L, typename R>
    requires std::is_base_of_v<tags::binary_op, Tag>
  std::ostream &operator<<(std::ostream &sout, expr<Tag, L, R> const &ex) {
    return sout << "(" << std::get<0>(ex.childs) << " " << Tag::name() << " " << std::get<1>(ex.childs) << ")";
  }

  /**
   * @brief Print an nda::clef::tags::if_else expression to std::ostream.
   *
   * @tparam C Type of the condition expression.
   * @tparam A Type of the return type when the condition is true.
   * @tparam B Type of the return type when the condition is false.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename C, typename A, typename B>
  std::ostream &operator<<(std::ostream &sout, expr<tags::if_else, C, A, B> const &ex) {
    return sout << "(" << std::get<0>(ex.childs) << " ? " << std::get<1>(ex.childs) << " : " << std::get<2>(ex.childs) << ")";
  }

  /**
   * @brief Print an nda::clef::tags::function expression to std::ostream.
   *
   * @tparam Ts Types of the arguments of the function.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename... Ts>
  std::ostream &operator<<(std::ostream &sout, expr<tags::function, Ts...> const &ex) {
    sout << "lambda"
         << "(";
    print_tuple(sout, ex.childs);
    return sout << ")";
  }

  /**
   * @brief Print a general nda::clef::tags::subscript expression to std::ostream.
   *
   * @tparam Ts Types of the subscript arguments.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename... Ts>
  std::ostream &operator<<(std::ostream &sout, expr<tags::subscript, Ts...> const &ex) {
    sout << "lambda"
         << "[";
    print_tuple(sout, ex.childs);
    return sout << "]";
  }

  /**
   * @brief Print an nda::clef::tags::terminal expression to std::ostream.
   *
   * @tparam T Type of the terminal child node.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::terminal, T> const &ex) {
    return sout << std::get<0>(ex.childs);
  }

  /**
   * @brief Print an nda::clef::tags::subscript expression to std::ostream.
   *
   * @tparam T Type of the child node.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::subscript, T> const &ex) {
    return sout << std::get<0>(ex.childs) << "[" << std::get<1>(ex.childs) << "]";
  }

  /**
   * @brief Print an nda::clef::tags::negate expression to std::ostream.
   *
   * @tparam T Type of the child node.
   * @param sout std::ostream object to print to.
   * @param ex nda::clef::expr object to print.
   * @return Reference to the std::ostream.
   */
  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::negate, T> const &ex) {
    return sout << "-(" << std::get<0>(ex.childs) << ")";
  }

  /**
   * @brief Print an nda::clef::make_fun_impl object to std::ostream.
   *
   * @tparam Expr Type of the expression.
   * @tparam Is Integer labels of the placeholders in the expression.
   * @param sout std::ostream object to print to.
   * @param f nda::clef::make_fun_impl object to print.
   * @return Reference to the std::ostream.
   */
  template <typename Expr, int... Is>
  std::ostream &operator<<(std::ostream &sout, make_fun_impl<Expr, Is...> const &f) {
    sout << "lazy function : (";
    variadic_print(sout, placeholder<Is>()...);
    return sout << ") --> " << f.ex;
  }

  /** @} */

} // namespace nda::clef
