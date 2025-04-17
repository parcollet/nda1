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
 * @brief Provides functionality to evaluate lazy expressions from the clef library.
 */

#pragma once

#include "./expression.hpp"
#include "./operation.hpp"
#include "./placeholder.hpp"
#include "./function.hpp"
#include "./utils.hpp"
#include "../macros.hpp"

#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <iostream>

namespace nda::clef {

  // explicitly list the types for which eval is not trivial
  template <typename T>
  constexpr bool eval_pass_through = not is_lazy<T>; // true;
  // template <int N>
  // constexpr bool eval_pass_through<placeholder<N>> = false;
  // template <typename... U>
  // constexpr bool eval_pass_through<expr<U...>> = false;
  template <typename T>
  constexpr bool eval_pass_through<std::reference_wrapper<T>> = false;
  // template <typename T, auto... Is>
  // constexpr bool eval_pass_through<function<T, Is...>> = false;

  /**
   * @addtogroup clef_eval
   * @{
   */
  // ----------   eval  : forward decl  -------------

  template <typename T>
  FORCEINLINE decltype(auto) eval1(T &&x, auto const &...pairs);

  template <typename T>
  FORCEINLINE decltype(auto) eval(T &&x, auto const &...)
    requires(eval_pass_through<std::decay_t<T>>)
  {
    return std::forward<T>(x);
  }
  // --------- placeholder --------------

  template <int N, typename... Pairs>
  FORCEINLINE decltype(auto) eval(placeholder<N>, Pairs const &...pairs) {

    // Position of the pair which contains N or -1
    constexpr int N_position = []<size_t... Ps>(std::index_sequence<Ps...>) {
      return ((Pairs::p == N ? int(Ps) + 1 : 0) + ...) - 1;
    }(std::make_index_sequence<sizeof...(Pairs)>{});

    if constexpr (N_position == -1) { // N is not one of the Is
      return placeholder<N>{};
    } else { // N is one of the Is
      auto &pair_N = std::get<N_position>(std::tie(pairs...));
      // FIXME in C++26
      // auto & pair_N = pairs...[N_position];
      // the pair is a temporary constructed for the time of the eval call
      // if it holds a reference, we return it, else we move the rhs object out of the pair
      if constexpr (std::is_lvalue_reference_v<decltype(pair_N.rhs)>) {
        return pair_N.rhs;
      } else {
        //std::cout << " MAKE?Ing COPY\n";
        //return std::move(pair_N.rhs);
        return auto{pair_N.rhs}; // make a copy
        //return (pair_N.rhs); // make a copy
      }
    }
  }

  // -----------------------

  template <typename T>
  FORCEINLINE decltype(auto) eval(std::reference_wrapper<T> const &wrapper, auto const &...pairs) {
    return eval(wrapper.get(), pairs...);
  }
  // ---------- expr -------------

  template <typename Tag, typename... Childs>
  FORCEINLINE decltype(auto) eval(expr<Tag, Childs...> const &ex, auto const &...pairs) {

    // can make the + with multiple arguments now !
    //return (+ eval(childs, pairs...) ...);

    //if constexpr (Kind == Addition)
    //     return eval(ex.childs...[0], pairs...) + eval(ex.childs...[1], pairs...);
    if constexpr (std::is_same_v<Tag, tags::plus>)
      return eval(std::get<0>(ex.childs), pairs...) + eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::minus>)
      return eval(std::get<0>(ex.childs), pairs...) - eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::multiplies>)
      return eval(std::get<0>(ex.childs), pairs...) * eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::divides>)
      return eval(std::get<0>(ex.childs), pairs...) / eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::eq>)
      return eval(std::get<0>(ex.childs), pairs...) == eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::leq>)
      return eval(std::get<0>(ex.childs), pairs...) <= eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::less>)
      return eval(std::get<0>(ex.childs), pairs...) < eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::greater>)
      return eval(std::get<0>(ex.childs), pairs...) > eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::geq>)
      return eval(std::get<0>(ex.childs), pairs...) >= eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::unaryplus>)
      return +eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::negate>)
      return -eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (std::is_same_v<Tag, tags::loginot>)
      return !eval(std::get<0>(ex.childs), pairs...);
    else
      // if_else, function, subscript require some more logic
      // C++26
      // auto &&[F, ...args] = ex.childs;
      // if contexpr ((Lazy<decltype(eval(args, pairs...)> or ...))
      //   return expr{tags::function{}, F, args...};
      // else return F(eval(args, pairs...)...);
      // 
      // return detail::operation<Tag>::invoke(eval(child, pairs...)...);
      return [&]<size_t... Is>(std::index_sequence<Is...>) __attribute__((always_inline))->decltype(auto) { // invoke call/subscript
        return detail::operation<Tag>::invoke(eval(std::get<Is>(ex.childs), pairs...)...);
      }
    (std::make_index_sequence<sizeof...(Childs)>{});
  }

  // ---------- function -------------

  //Evaluates the underlying expression and rebuild the function.
  template <typename T, int... Is, typename... Pairs>
  FORCEINLINE decltype(auto) eval(function<T, Is...> const &f, Pairs const &...pairs) {
    // makes no sense if some of the Pairs placeholders are included in the Is.
    constexpr uint64_t I = ((1ull << Is) + ...);
    constexpr uint64_t J = ((1ull << Pairs::p) + ...);
    static_assert((I & J) == 0, "Impossible evaluation. You can not evaluate a function on the placeholders used to define the function");
    return make_function(eval(f.ex, pairs...), placeholder<Is>{}...);
  }

  // -----------  eval ------------
  // The user facing function

  /**
   * @brief Evaluate expression on pairs (placeholder = value)
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<1> j_;
   * auto ex = i_ + j_;
   * auto res = nda::clef::eval(ex, i_ = 1, j_ = 2); // int res = 3;
   * @endcode
   *
   * If x is 
   *    a clef::expr, replace the placeholders by their value and recompute the expression.
   *        if all the placeholders are specified (full evaluation), return the result. 
   *        Otherwise (partial evaluation), return another clef::expr<> with only the remaining placeholders.
   *    a function, evaluate the underlying expression and reconstructs a new function from the expression.
   *      FIXME: EXAMPLE
   *    anything else, pass x through.
   *
   * @tparam T Any type
   * @tparam Pairs Must be nda::clef::pair<...>
   * @param x Expression/object to be evaluated. 
   * @param pairs of (placeholder, value) 
   * @return Cf below.
   */
  template <typename T>
  FORCEINLINE decltype(auto) eval1(T &&x, auto const &...pairs) {
    if constexpr (requires { eval_impl(std::forward<T>(x), pairs...); })
      return eval_impl(std::forward<T>(x), pairs...);
    else
      return std::forward<T>(x);
  }
  /** @} */

  /*

 - expr_with_context : expr + tuple of objects.
 - eval :
     - r = eval(...) : put into  dangling_ref<n_pair, T> -> cast into a T&

     - Compute the list [pair_pos] for all the dangling_ref

     - list [ (pair_idx, new_idx)] from [0/1] for each pair, order on pair_idx
     - a consteval function get_new_idx(pair_idx) -> new_idx: search
     - [n_pair of dangling ref] + [0/1] ---> [new index for each pair] -> consteval fun
     - rebuild the expression replacing the dangling_ref<I, T> -> std::get< get_new_idx(I)>(context);
     - MUST PASS the context to the evaluator !!!   eval(x, context, pairs ...)
        - in general : pass empty tuple. ---> why not the empty tuple in all expression ? 


    1- dangling_ref<I,T>
    2- [0/1] for each pair
    2- [new_idx for each pair pos] : from [0/1] -> accumulate
    3- Context = get the temporaries and move them
    4- eval : ph when pair_pos is 1, use a dangling_ref <new_idx>
    2- change the eval_impl to accept a context.
    3- Add eval dangling_ref<I, T> -> using the context
    4- One pass only. Now if we want to minimize the Context tuple, we need another pass.

*/

} // namespace nda::clef
