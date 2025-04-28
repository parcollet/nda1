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
#include <iostream>
#include <fmt/core.h>
#include "./placeholder.hpp"
#include "./expression.hpp"
#include "./function.hpp"

namespace nda::clef {

  /**
   * @addtogroup clef_utils
   * @{
   */

  /**
   * @brief Overloads the stream insertion operator (<<) for nda::clef::placeholder.
   *
   * This function allows an nda::clef::placeholder object to be directly inserted into an
   * output stream. It formats the placeholder as "_N", where N is its integer label.
   *
   * @tparam N The integer label of the placeholder.
   * @param sout The output stream to which the placeholder will be inserted.
   * @return std::ostream& A reference to the output stream after insertion.
   */
  template <int N>
  std::ostream &operator<<(std::ostream &sout, placeholder<N>) {
    return sout << "_" << N;
  }

  /**
   * @brief Overloads the stream insertion operator (<<) for std::reference_wrapper.
   * 
   * This function allows a std::reference_wrapper<T> to be directly inserted into an
   * output stream. It inserts its underlying reference  into the stream.
   * 
   * @tparam T The type of the object wrapped by std::reference_wrapper.
   * @param sout The output stream to which the wrapped object will be inserted.
   * @param wrapper The std::reference_wrapper containing the object to be inserted.
   * @return std::ostream& A reference to the output stream after insertion.
   */
  template <typename T>
  std::ostream &operator<<(std::ostream &sout, std::reference_wrapper<T> const &wrapper) {
    return sout << wrapper.get();
  }

  /**
   * Overloads the stream insertion operator (<<) to provide a formatted
   * string representation of an `expr` object based on its type and structure.
   *
   * @param sout The output stream to write to.
   * @param ex The expression object to be formatted and written to the stream.
   * @return A reference to the output stream after writing the formatted expression.
   */
  template <NodeKind K, typename... T>
  std::ostream &operator<<(std::ostream &sout, expr<K, T...> const &ex) {

    auto call_printer = [&](char opener, char closer) -> decltype(auto) {
      auto print = [i = 0, &sout, opener](auto const &x) mutable -> void {
        if (i == 1) sout << opener;
        if (i > 1) sout << ", ";
        ++i;
        if constexpr (requires { sout << x; })
          sout << x;
        else
          sout << "[??]";
      };
      // FIXME C++26. Simply say
      // auto &[... x] = t;
      // (print(x),...);
      [&]<size_t... Is>(std::index_sequence<Is...>) { (print(std::get<Is>(ex.childs)), ...); }(std::make_index_sequence<sizeof...(T)>{});
      return sout << closer;
    };

    if constexpr (K == Call) {
      return call_printer('(', ')');
    } //
    else if constexpr (K == Subscript) {
      return call_printer('[', ']');
    } //
    else if constexpr (sizeof...(T) == 1) {
      auto &[arg0] = ex.childs;
      if constexpr ((K == Leaf) || (K == UnaryPlus))
        return sout << fmt::format("({})", arg0);
      else if constexpr (K == Negate)
        return sout << fmt::format("(- {})", arg0);
      else
        static_assert(false, "Unknown unary operation");
    } //
    else if constexpr (sizeof...(T) == 2) {
      auto &[arg0, arg1] = ex.childs;
      auto pr            = [&](const char *op) -> decltype(auto) { return sout << arg0 << " " << op << " " << arg1; };
      if constexpr (K == Add)
        return pr("+");
      else if constexpr (K == Sub)
        return pr("-");
      else if constexpr (K == Mul)
        return pr("*");
      else if constexpr (K == Div)
        return pr("/");
      else if constexpr (K == Eq)
        return pr("==");
      else if constexpr (K == Less)
        return pr("<");
      else if constexpr (K == Greater)
        return pr(">");
      else if constexpr (K == Leq)
        return pr("<=");
      else if constexpr (K == Geq)
        return pr(">=");
      else
        static_assert(0, "Unknown binary operation");
    } //
    else if constexpr (sizeof...(T) == 3) {
      auto &[arg0, op, arg1] = ex.childs;
      return sout << arg0 << " ? " << op << " : " << arg1;
    } else
      static_assert(false, "Unknown expression type");
  }

  /**
   * @brief Print an nda::clef::function object to std::ostream.
   */
  template <typename Expr, int I0, int... Is>
  std::ostream &operator<<(std::ostream &sout, function<Expr, I0, Is...> const &f) {
    sout << "[(" << placeholder<I0>{};
    (void(sout << ", " << placeholder<Is>{}), ...);
    return sout << ") --> " << f.ex << ']';
  }

  /** @} */

} // namespace nda::clef
