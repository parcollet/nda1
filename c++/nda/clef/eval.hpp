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
#include <tuple>

#include "./expression.hpp"
#include "./operation.hpp"
#include "./placeholder.hpp"
#include "./function.hpp"

namespace nda::clef {

  // This is to write a simpler eval function, in order to
  // minimize the call stack depth in error messages.
  //
  // eveything which is lazy needs to be evaluated, the rest is passed through during an eval.
  /// Explicitly list the types for which eval is not trivial
  template <typename T>
  constexpr bool eval_pass_through = not is_lazy<T>;

  // the reference wrapper is a special case, we need to eval it to unwrap it
  template <typename T>
  constexpr bool eval_pass_through<std::reference_wrapper<T>> = false;

  /**
   * @addtogroup clef_eval
   * @{
   */

  // FIXME : shall we put the doxgen doc here in a ifdef for doxygen only ?
  // eval function is of the form
  // with x is anything, and pair is pack of ph_value_pair
  // FORCEINLINE decltype(auto) eval(auto &&x, auto const &...pairs);

  // ----------  Default case : do nothing, pass through -------------

  template <typename T>
    requires(eval_pass_through<std::decay_t<T>>) // beware of the decay...
  FORCEINLINE decltype(auto) eval(T &&x, auto const &...) {
    return std::forward<T>(x);
  }
  // --------- eval a placeholder --------------

  template <int N, typename... Pairs>
  FORCEINLINE decltype(auto) eval(placeholder<N>, Pairs const &...pairs) {

    // Position of the pair which contains N or -1
    constexpr int N_position = []() { // a compile time computation !
      int pos = 0;
      for (auto i : {Pairs::idx...}) {
        if (i == N) return pos;
        ++pos;
      }
      return -1;
    }();

    if constexpr (N_position == -1) { // N is not one of the Is
      return placeholder<N>{};        // do nothing, just pass through the placeholder.
    } else {                          // N is one of the Is
      // auto & pair_N = pairs...[N_position];  // C++26
      auto &pair_N = std::get<N_position>(std::tie(pairs...)); // C++23
      // the pair is a temporary constructed for the time of the eval call
      // if it holds a reference, we return it, else we COPY its value.
      // WE CAN NOT MOVE IT out as there maybe several identical placeholder in the tree.
      if constexpr (std::is_lvalue_reference_v<decltype(pair_N.value)>)
        return pair_N.value;
      else
        return auto{pair_N.value}; // make a copy
    }
  }

  // ----------- reference_wrapper ------------

  template <typename T>
  FORCEINLINE decltype(auto) eval(std::reference_wrapper<T> const &wrapper, auto const &...pairs) {
    return eval(wrapper.get(), pairs...);
  }
  // ---------- expr -------------

  template <NodeKind K, typename... Childs>
  FORCEINLINE decltype(auto) eval(expr<K, Childs...> const &ex, auto const &...pairs) {
    if constexpr (K == Add)
      // return eval(ex.childs...[0], pairs...) + eval(ex.childs...[1], pairs...); // C++26
      return eval(std::get<0>(ex.childs), pairs...) + eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Sub)
      return eval(std::get<0>(ex.childs), pairs...) - eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Mul)
      return eval(std::get<0>(ex.childs), pairs...) * eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Div)
      return eval(std::get<0>(ex.childs), pairs...) / eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Eq)
      return eval(std::get<0>(ex.childs), pairs...) == eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Leq)
      return eval(std::get<0>(ex.childs), pairs...) <= eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Less)
      return eval(std::get<0>(ex.childs), pairs...) < eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Greater)
      return eval(std::get<0>(ex.childs), pairs...) > eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == Geq)
      return eval(std::get<0>(ex.childs), pairs...) >= eval(std::get<1>(ex.childs), pairs...);
    else if constexpr (K == UnaryPlus)
      return +eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (K == Negate)
      return -eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (K == Loginot)
      return !eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (K == Leaf)
      return eval(std::get<0>(ex.childs), pairs...);
    else if constexpr (K == IfElse) {
      return eval(std::get<0>(ex.childs), pairs...) ? eval(std::get<1>(ex.childs), pairs...) : eval(std::get<2>(ex.childs), pairs...);
    } //
    else if constexpr ((K == Call) || (K == Subscript)) {
      // auto &&[...ch] = ex.childs; // C++26
      // return detail::operation<K>::invoke(eval(ch, pairs...)...); // C++26
      // Or replace the operation
      return [&]<auto... Is>(std::index_sequence<Is...>) __attribute__((always_inline)) -> decltype(auto) { // invoke call/subscript
        return detail::operation<K>::invoke(eval(std::get<Is>(ex.childs), pairs...)...);
      }(std::make_index_sequence<sizeof...(Childs)>{});
    } else
      static_assert(false, "Unknown expression kind in eval !");
  }

  // ---------- function -------------

  //Evaluates the underlying expression and rebuild the function.
  template <typename T, int... Is, typename... Pairs>
  FORCEINLINE decltype(auto) eval(function<T, Is...> const &f, Pairs const &...pairs) {
    // makes no sense if some of the Pairs placeholders are included in the Is.
    constexpr uint64_t I = ((1ull << Is) + ...);
    constexpr uint64_t J = ((1ull << Pairs::idx) + ...);
    static_assert((I & J) == 0, "Impossible evaluation. You can not evaluate a function on the placeholders used to define the function");
    return function{eval(f.ex, pairs...), placeholder<Is>{}...};
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
   * auto res = eval(ex, i_ = 1, j_ = 2); // int res = 3;
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
