//  File is generated by vim.
//   To regenerate the file, use in this buffer the vim script
//
//   :source c++/nda/matrix.vim
#pragma once
#include "./matrix_view.hxx"

namespace nda {

  // UNCOMMENT_FOR_MATRIX
  /// Class template argument deduction
  template <typename T>
  matrix(T)->matrix<get_value_t<std::decay_t<T>>>;

  // DELETED_CODE

  // ---------------------- matrix--------------------------------

  template <typename ValueType, uint64_t StrideOrder>
  class matrix {
    static_assert(!std::is_const<ValueType>::value, "ValueType can not be const. WHY ?");

    public:
    ///
    using value_t = ValueType;
    ///
    using regular_t = matrix<ValueType>;
    ///
    using view_t = matrix_view<ValueType, layout_info_e::contiguous, StrideOrder>;
    ///
    using const_view_t = matrix_view<ValueType const, layout_info_e::contiguous, StrideOrder>;

    using storage_t = mem::heap::handle<ValueType>;
    using idx_map_t = idx_map<2, StrideOrder, layout_info_e::contiguous>;

    //    static constexpr uint64_t stride_order = StrideOrder;
    //  static constexpr layout_info_e stride_order= layout_info_e::contiguous;

    static constexpr int rank      = 2;
    static constexpr bool is_const = false;
    static constexpr bool is_view  = false;

    private:
    template <typename IdxMap>
    using my_view_template_t = matrix_view<value_t, IdxMap::layout_info, permutations::encode(IdxMap::stride_order)>;

    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Empty matrix
    matrix() = default;

    /// Makes a deep copy, since matrix is a regular type
    matrix(matrix const &x) : _idx_m(x.indexmap()), _storage(x.storage()) {}

    ///
    matrix(matrix &&X) = default;

    /** 
     * Construct with a shape [i0, is ...]. 
     * Int must be convertible to long, and there must be exactly 2 arguments.
     * 
     * @param i0, is ... lengths in each dimensions
     * @example matrix_constructors
     */
    template <typename... Int>
    explicit matrix(long i0, Int... is) {
      static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(Int) + 1 == 2, "Incorrect number of arguments : should be exactly 2. ");
      _idx_m   = idx_map_t{{i0, is...}};
      _storage = storage_t{_idx_m.size()};
      // It would be more natural to construct _idx_m, storage from the start, but the error message in case of false # of parameters (very common)
      // is better like this. FIXME to be tested in benchs
    }

    /** 
     * Construct with the given shape
     * 
     * @param shape  Shape of the matrix (lengths in each dimension)
     */
    explicit matrix(shape_t<2> const &shape) : _idx_m(shape), _storage(_idx_m.size()) {}

    /** 
     * [Advanced] Construct from an indexmap and a storage handle.
     *
     * @param idxm index map
     * @param mem_handle  memory handle
     * NB: make a new copy.
     */
    //template <char RBS>
    //matrix(idx_map<2> const &idxm, mem::handle<ValueType, RBS> mem_handle) : _idx_m(idxm), _storage(std::move(mem_handle)) {}

    /// Construct from anything that has an indexmap and a storage compatible with this class
    //template <typename T> matrix(T const &a) REQUIRES(XXXX): matrix(a.indexmap(), a.storage()) {}

    /** 
     * From any type modeling NdArray
     * Constructs from x.shape() and then assign from the evaluation of x.
     * 
     * @tparam A A type modeling NdArray
     * @param x 
     */
    template <typename A>
    matrix(A const &x) REQUIRES(is_ndarray_v<A>) : matrix{x.shape()} {
      static_assert(std::is_convertible_v<get_value_t<A>, value_t>,
                    "Can not construct the matrix. ValueType can not be constructed from the value_t of the argument");
      nda::details::assignment(*this, x);
    }

    /** 
     * [Advanced] From a shape and a storage handle (for reshaping)
     * NB: make a new copy.
     *
     * @param shape  Shape of the matrix (lengths in each dimension)
     * @param mem_handle  memory handle
     */
    //template <char RBS>
    //matrix(shape_t<2> const &shape, mem::handle<ValueType, RBS> mem_handle) : matrix(idx_map_t{shape}, mem_handle) {}

    // --- with initializers

    // DELETED_CODE

    private: // impl. detail for next function
    template <typename T>
    static shape_t<2> _comp_shape_from_list_list(std::initializer_list<std::initializer_list<T>> const &ll) {
      long s = -1;
      for (auto const &l1 : ll) {
        if (s == -1)
          s = l1.size();
        else if (s != l1.size())
          throw std::runtime_error("initializer list not rectangular !");
      }
      return {long(ll.size()), s};
    }

    public:
    /**
     * Construct from the initializer list of list 
     *
     * @tparam T Any type from which ValueType is constructible
     * @param ll Initializer list of list
     * @requires  ValueType is constructible from T
     */
    template <typename T>
    matrix(std::initializer_list<std::initializer_list<T>> const &ll) //
       REQUIRES((2 == 2) and std::is_constructible_v<value_t, T>)
       : matrix(_comp_shape_from_list_list(ll)) {
      long i = 0, j = 0;
      for (auto const &l1 : ll) {
        for (auto const &x : l1) { (*this)(i, j++) = x; }
        j = 0;
        ++i;
      }
    }

    /**
     * [Advanced] Construct from shape and a Lambda to initialize the elements. 
     * a(i,j,k,...) is initialized to initializer(i,j,k,...) at construction.
     * Specially useful for non trivially constructible type
     *
     * @tparam Initializer  a callable on 2 longs which returns something is convertible to ValueType
     * @param shape  Shape of the matrix (lengths in each dimension)
     * @param initializer The lambda
     */
    template <typename Initializer>
    explicit matrix(shape_t<2> const &shape, Initializer initializer)
       REQUIRES(details::_is_a_good_lambda_for_init<ValueType, Initializer>(std::make_index_sequence<2>()))
       : _idx_m(shape), _storage{_idx_m.size(), mem::do_not_initialize} {
      nda::for_each(_idx_m.lengths(), [&](auto const &... x) { _storage.init_raw(_idx_m(x...), initializer(x...)); });
    }

    //------------------ Assignment -------------------------

    ///
    matrix &operator=(matrix &&x) = default;

    /// Deep copy (matrix is a regular type). Invalidates all references to the storage.
    matrix &operator=(matrix const &X) = default;

    /** 
     * Resizes the matrix (if necessary).
     * Invalidates all references to the storage.
     *
     * @tparam RHS A scalar or an object modeling NdArray
     */
    template <typename RHS>
    matrix &operator=(RHS const &rhs) {
      static_assert(is_ndarray_v<RHS> or is_scalar_for_v<RHS, matrix>, "Assignment : RHS not supported");
      if constexpr (is_ndarray_v<RHS>) resize(rhs.shape());
      nda::details::assignment(*this, rhs);
      return *this;
    }

    //------------------ resize  -------------------------
    /** 
     * Resizes the matrix.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     * @tparam Int Integer type
     * @param i0 New dimension
     * @param is New dimension
     */
    template <typename... Int>
    void resize(long i0, Int const &... is) {
      static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(is) + 1 == 2, "Incorrect number of arguments for resize. Should be 2");
      static_assert(std::is_copy_constructible_v<ValueType>, "Can not resize an matrix if its value_t is not copy constructible");
      resize(shape_t<2>{i0, is...});
    }

    /** 
     * Resizes the matrix.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     * @param shape  New shape of the matrix (lengths in each dimension)
     */
    [[gnu::noinline]] void resize(shape_t<2> const &shape) {
      _idx_m = idx_map_t(shape);
      // Construct a storage only if the new index is not compatible (size mismatch).
      if (_storage.size() != _idx_m.size()) _storage = mem::handle<ValueType, 'R'>{_idx_m.size()};
    }

    // --------------------------

#include "./_regular_view_common.hpp"
  };

} // namespace nda
