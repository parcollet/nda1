// Copyright (c) 2018-2020 Simons Foundation
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

#pragma once
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <numeric>

#include "../macros.hpp"
#include "../traits.hpp"
#include "./range.hpp"
#include "./bound_check_worker.hpp"
#include "./for_each.hpp"

namespace nda {

  // forward
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map;

  namespace slice_static {
    template <int R, uint64_t SE, uint64_t SO, layout_prop_e LP, typename... T>
    FORCEINLINE decltype(auto) slice_stride_order(idx_map<R, SE, SO, LP> const &idxm, T const &...x);
  }
  // end forward

  template <int Rank>
  constexpr uint64_t Fortran_stride_order = nda::encode(nda::permutations::reverse_identity<Rank>());

  template <int Rank>
  constexpr uint64_t C_stride_order = nda::encode(nda::permutations::identity<Rank>());

  // -----------------------------------------------------------------------------------
  /**
   *
   * The layout that maps the indices to linear index.
   *
   * Basically lengths and strides for each dimensions
   *
   * @tparam Rank : rank of the index map
   * 
   * @tparam StaticExtents : encoded std::array{0, d1, 0, d3}
   *   where d1, d3 are static dimensions for index 1,3
   *         NB Limitation : d1, d3 < 16 (until C++20)
   *         0 mean dynamic dimension
   *   NB : if StaticExtents ==0, it means all dimensions are static
   *
   * @tparam StrideOrder : a permutation for the memory stride_order of the array
   *    
   *    StrideOrder[0] : the slowest index,
   *    StrideOrder[Rank-1] : the fastest index
   *    Example :
   *        012 : C the last index is the fastest
   *        210 : Fortran, the first index is the fastest
   *        120 : storage (i,j,k) is : index j is slowest, then k, then i
   *    
   *    NB : StrideOrder = 0 is the default and it is means 0 order
   * 
   * @tparam LayoutProp : A flags of compile time guarantees for the layout
   *
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map {
    static_assert(Rank < 16, "Rank must be < 16"); // C++17 constraint. Relax this in C++20
    static_assert((StrideOrder != 0) or (Rank == 1), "Oops");
    std::array<long, Rank> len, str; // lenghts and strides

    public:
    static constexpr uint64_t static_extents_encoded      = StaticExtents;
    static constexpr std::array<int, Rank> static_extents = decode<Rank>(StaticExtents);

    static constexpr std::array<int, Rank> stride_order =
       (StrideOrder == 0 ? permutations::identity<Rank>() : decode<Rank>(StrideOrder)); // 0 is C stride_order
    static constexpr uint64_t stride_order_encoded = encode(stride_order);

    static constexpr layout_prop_e layout_prop = LayoutProp;
    static constexpr layout_info_t layout_info = layout_info_t{stride_order_encoded, layout_prop};

    template <typename T>
    static constexpr int argument_is_allowed_for_call = std::is_constructible_v<long, T>;

    template <typename T>
    static constexpr int argument_is_allowed_for_call_or_slice =
       std::is_same_v<range, T> or std::is_same_v<range::all_t, T> or std::is_same_v<ellipsis, T> or std::is_constructible_v<long, T>;

    protected:
    static constexpr int n_dynamic_extents = []() {
      int r = 0;
      for (int u = 0; u < Rank; ++u) r += (static_extents[u] == 0 ? 1 : 0);
      return r;
    }();

    public:
    // ----------------  Accessors -------------------------

    /// Rank of the map (number of arguments)
    static constexpr int rank() noexcept { return Rank; }

    /// Total number of elements (products of lengths in each dimension).
    // NB recomputed at each call (FIXME Optimize this ?)
    [[nodiscard]] long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1, std::multiplies<>{}); }

    /// Compile time size, 0 means "dynamical"
    static constexpr long ce_size() noexcept {
      if constexpr (n_dynamic_extents != 0) { // quick general case
        return 0;
      } else {
        long s = 1;
        for (int u = 0; u < Rank; ++u) s *= static_extents[u];
        return s;
      }
    }

    /// Lengths of each dimension.
    [[nodiscard]] std::array<long, Rank> const &lengths() const noexcept { return len; }

    /// Strides of each dimension.
    [[nodiscard]] std::array<long, Rank> const &strides() const noexcept { return str; }

    /// Value of the minimum stride (i.e the fastest one)
    [[nodiscard]] long min_stride() const noexcept { return str[stride_order[Rank - 1]]; }

    /// Is the data contiguous in memory ? [NB recomputed at each call]
    [[nodiscard]] bool is_contiguous() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size());
    }

    /// Is the data strided 1d in memory ? [NB recomputed at each call]
    [[nodiscard]] bool is_strided_1d() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size() * min_stride());
    }

    /// Is the order in memory C ?
    static constexpr bool is_stride_order_C() {
      // operator == of std:array is constexpr only since C++20
      //#if __cplusplus > 201703L
      //      return (stride_order == permutations::identity<Rank>());
      //#else
      return (encode(stride_order) == encode(permutations::identity<Rank>()));
      //#endif
    }

    /// Is the order in memory Fortran ?
    static constexpr bool is_stride_order_Fortran() {
      //#if __cplusplus > 201703L
      //      return (stride_order == permutations::reverse_identity<Rank>());
      //#else
      return (encode(stride_order) == encode(permutations::reverse_identity<Rank>()));
      //#endif
    }

    // ----------------  Constructors -------------------------

    private:
    // compute strides for a contiguous array from len
    void compute_strides_contiguous() {
      long s = 1;
      for (int v = rank() - 1; v >= 0; --v) { // rank() is constexpr, allowing compiler to transform loop...
        int u  = stride_order[v];
        str[u] = s;
        s *= len[u];
      }
      ENSURES(s == size());
    }

    // FIXME ADD A CHECK layout_prop_e ... compare to stride and

    public:
    /// Default constructor. Strides are not initiliazed.
    idx_map() {
      if constexpr (n_dynamic_extents == 0) { // full static array
        for (int u = 0; u < Rank; ++u) len[u] = static_extents[u];
        compute_strides_contiguous();
      } else {
        for (int u = 0; u < Rank; ++u)
          len[u] = 0; // FIXME. Needed ? To have the proper invariant of the array : shape = (0,0,...) and pointer is null
      }
    }

    idx_map(idx_map const &) = default;
    idx_map(idx_map &&)      = default;
    idx_map &operator=(idx_map const &) = default;
    idx_map &operator=(idx_map &&) = default;

    private:
    [[nodiscard]] bool strides_compatible_to_stride_order() const {
      for (int u = 0; u < Rank - 1; ++u)
        if (str[stride_order[u]] < str[stride_order[u + 1]]) return false;
      return true;
    }

    public:
    /** 
     * From an idxmap with other info flags
     * @param idxm
     */
    template <layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents, StrideOrder, P> const &idxm) noexcept : len(idxm.lengths()), str(idxm.strides()) {
      EXPECTS(strides_compatible_to_stride_order());
      if constexpr (not layout_property_compatible(P, LayoutProp)) {
        if constexpr (has_contiguous(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_contiguous(),
             "Failed check of contiguity. Constructing a contiguous layout from another layout which was not guaranteed to be contiguous at compile time. The check fails so your program is incorrect");
        }
        if constexpr (has_strided_1d(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_strided_1d(),
             "Failed check of quasi-contiguity (1d-strided). Constructing a contiguous layout from another layout which was not guaranteed to be quasi-contiguous at compile time. The check fails so your program is incorrect");
        }
      }
    }

    private:
    void assert_static_extents_and_len_are_compatible() const {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if constexpr (n_dynamic_extents != Rank) { // there are some static extents
// to avoid warning
#ifndef NDEBUG
        for (int u = 0; u < Rank; ++u)
          if (static_extents[u] != 0) EXPECTS(static_extents[u] == len[u]);
#endif
      }
#endif
    }

    public:
    /// Construct from a compatible static_extents
    template <uint64_t SE, layout_prop_e P>
    idx_map(idx_map<Rank, SE, StrideOrder, P> const &idxm) noexcept(false) : len(idxm.lengths()), str(idxm.strides()) { // can throw
      EXPECTS(strides_compatible_to_stride_order());
      if constexpr (not layout_property_compatible(P, LayoutProp)) {
        if constexpr (has_contiguous(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_contiguous(),
             "Failed check of contiguity. Constructing a contiguous layout from another layout which was not guaranteed to be contiguous at compile time. The check fails so your program is incorrect");
        }
        if constexpr (has_strided_1d(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_strided_1d(),
             "Failed check of quasi-contiguity (1d-strided). Constructing a contiguous layout from another layout which was not guaranteed to be quasi-contiguous at compile time. The check fails so your program is incorrect");
        }
      }
      assert_static_extents_and_len_are_compatible();
    }

    ///
    idx_map(std::array<long, Rank> const &shape, std::array<long, Rank> const &strides) : len(shape), str(strides) {
      if (not strides_compatible_to_stride_order()) throw std::runtime_error("ERROR: strides of idx_map do not match stride order of the type\n");
    }

    /// Construct from the shape. If StaticExtents are present, the corresponding component of the shape must be equal to it.
    idx_map(std::array<long, Rank> const &shape) noexcept : len(shape) {
      assert_static_extents_and_len_are_compatible();
      compute_strides_contiguous();
    }

    private:
    static std::array<long, Rank> merge_static_and_dynamic_extents(std::array<long, n_dynamic_extents> const &s) {
      std::array<long, Rank> r;
      for (int u = 0, v = 0; u < Rank; ++u) r[u] = (static_extents[u] == 0 ? s[v++] : static_extents[u]);
      return r;
    }

    public:
    /// When StaticExtents are present, constructs from the dynamic extents only
    idx_map(std::array<long, n_dynamic_extents> const &shape) noexcept REQUIRES((n_dynamic_extents != Rank) and (n_dynamic_extents != 0))
       : idx_map(merge_static_and_dynamic_extents(shape)) {}

    /// \private
    /// trap for error. If one tries to construct a view with a mismatch of stride order
    // The compiler selects this constructor instead of presenting a long list, and then goes into a dead end.
    template <uint64_t StaticExtents2, uint64_t StrideOrder2, layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents2, StrideOrder2, P> const &) REQUIRES(StrideOrder != StrideOrder2) {
      static_assert((StrideOrder == StrideOrder2), "Can not construct a layout from another one with a different stride order");
    }

    /// \private
    /// trap for error. For R = Rank, the non template has priority
    template <int R>
    idx_map(std::array<long, R> const &) REQUIRES(R != Rank) {
      static_assert(R == Rank, "Rank of the argument incorrect in idx_map construction");
    }

    // ----------------  Call operator -------------------------
    private:
    template <bool skip_stride, auto Is>
    [[nodiscard]] FORCEINLINE long myget(ellipsis) const noexcept {
      return 0;
    }

    template <bool skip_stride, auto Is>
    [[nodiscard]] FORCEINLINE long myget(long arg) const noexcept {
      if constexpr (skip_stride and (Is == stride_order[Rank - 1])) // this is the slowest stride
        return arg;
      else
        return arg * std::get<Is>(str);
    }

    static constexpr bool smallest_stride_is_one = has_smallest_stride_is_one(LayoutProp);

    // call implementation
    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const noexcept {
      static constexpr int e_pos = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...) - 1; // position ellipsis + 1 or 0

      if constexpr (e_pos == -1) { // common case, no ellipsis
        if constexpr (smallest_stride_is_one)
          return (myget<true, Is>(args) + ...);
        else
          return ((args * std::get<Is>(str)) + ...);
      } else {
        // there is an empty ellipsis to skip
        return (myget<smallest_stride_is_one, (Is < e_pos ? Is : Is - 1)>(args) + ...);
      }
    }

    public:
    /**
     * Number of variables must be exactly the rank or are optionally
     * checked at runtime
     *
     * @return : 
     *       the linear position 
     *
     */
    template <typename... Args>
    FORCEINLINE long operator()(Args const &...args) const
#ifdef NDA_ENFORCE_BOUNDCHECK
       noexcept(false) {
      details::assert_in_bounds(rank(), len.data(), args...);
#else
       noexcept(true) {
#endif
      // there may be an empty ellipsis which we will need to skip. e_pos = 128 if no ellipsis
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    // ----------------  Slice -------------------------

    template <typename... Args>
    auto slice(Args const &...args) const {
      return slice_static::slice_stride_order(*this, args...);
    }

    // ----------------  Comparison -------------------------

#if __cplusplus > 201703L
    bool operator==(idx_map const &x) const = default;
#else
    bool operator==(idx_map const &x) const { return (len == x.len) and (str == x.str); }
    bool operator!=(idx_map const &x) { return !(operator==(x)); }
#endif

    // ---------------- Transposition -------------------------

    template <uint64_t Permutation>
    auto transpose() const {

      static constexpr std::array<int, Rank> permu              = decode<Rank>(Permutation);
      static constexpr std::array<int, Rank> new_stride_order   = permutations::apply(permu, stride_order);
      static constexpr std::array<int, Rank> new_static_extents = permutations::apply_inverse(permu, static_extents);

      // Compute the new layout_prop of the new view
      // NB : strided_1d property is preserved, but smallest_stride_is_one is not
      // FIXME
      static constexpr layout_prop_e new_layout_prop =
         layout_prop_e::none; // BUT FIX (has_strided_1d(layout_prop) ? layout_prop_e::strided_1d : layout_prop_e::none);

      return idx_map<Rank, encode(new_static_extents), encode(new_stride_order), new_layout_prop>{permutations::apply_inverse(permu, lengths()),
                                                                                                  permutations::apply_inverse(permu, strides())};
    }

  }; // idx_map class

} // namespace nda
