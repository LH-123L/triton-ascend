# triton.language

## Ascend Extension API List

|api|Description|
|--|--|
|[extract_slice](./Extension_Ops/extract_slice.md)|  Extracts a tensor from the input tensor according to the offsets, sizes, and strides specified by the operation. |
|[insert_slice](./Extension_Ops/insert_slice.md)| Inserts a tensor (subtensor) into another tensor at the specified position, according to the offsets, sizes, and strides specified by the operation. |
|[sync_block](./Extension_Ops/sync_block.md) | An explicit inter-core synchronization instruction used to coordinate execution order and data consistency between different cores in the Cube-Vector architecture. |
|[compile_hint](./Extension_Ops/compile_hint.md) | A compiler hint mechanism that allows users to attach metadata to tensors, which is passed to the compiler backend to guide optimization and code generation.|
|[multibuffer](./Extension_Ops/multibuffer.md) | Sets up multi-buffering for a tensor, allowing the compiler to create multiple copies of the same tensor. |
|[parallel](./Extension_Ops/parallel.md) | `parallel` is an iterator specifically designed for multi-core parallel execution, providing explicit multi-core parallelism semantics. |
|[get_element](./Extension_Ops/get_element.md)| Reads a single element from the input tensor at the given index. |
|[index_select high-performance interface](./Extension_Ops/index_select_simd.md) | Parallelly gathers multiple indices along non-trailing axes and moves data from global memory (GM) directly to the correct positions in the unified buffer (UB) in units of tiles with zero copy. This operation is equivalent to a high-performance implementation of `torch.index_select`, suitable for scenarios such as embedding layer lookups and sparse index access. |

```{toctree}
:maxdepth: 3
:hidden:

Extension_Ops/extract_slice.md
Extension_Ops/insert_slice.md
Extension_Ops/sync_block.md
Extension_Ops/compile_hint.md
Extension_Ops/multibuffer.md
Extension_Ops/parallel.md
Extension_Ops/get_element.md
Extension_Ops/index_select_simd.md
```

## Atomic Operations

|api|Description|
|--|--|
|[atomic_add](./Atomic_Ops/atomic_add.md)  |Performs atomic addition at the memory location specified by pointer |
|[atomic_and](./Atomic_Ops/atomic_and.md)  |Performs an atomic logical AND at the memory location specified by pointer |
|[atomic_cas](./Atomic_Ops/atomic_cas.md)  |Performs an atomic compare-and-swap operation at the memory location specified by pointer |
|[atomic_max](./Atomic_Ops/atomic_max.md)  |Performs an atomic maximum operation at the memory location specified by pointer |
|[atomic_min](./Atomic_Ops/atomic_min.md)  |Performs an atomic minimum operation at the memory location specified by pointer |
|[atomic_or](./Atomic_Ops/atomic_or.md)  |Performs an atomic logical OR at the memory location specified by pointer |
|[atomic_xchg](./Atomic_Ops/atomic_xchg.md)  |Performs an atomic exchange operation at the memory location specified by pointer |
|[atomic_xor](./Atomic_Ops/atomic_xor.md)  |Performs an atomic logical XOR at the memory location specified by pointer |

```{toctree}
:maxdepth: 3
:hidden:

Atomic_Ops/atomic_add.md
Atomic_Ops/atomic_and.md
Atomic_Ops/atomic_cas.md
Atomic_Ops/atomic_max.md
Atomic_Ops/atomic_min.md
Atomic_Ops/atomic_or.md
Atomic_Ops/atomic_xchg.md
Atomic_Ops/atomic_xor.md
```

## Comparison Operations

|api|Description|
|--|--|
| [eq](./Comparing_Ops/eq.md) | Compares the elements of two tensors, equivalent to `==` |
| [le](./Comparing_Ops/le.md) | Compares the elements of two tensors, equivalent to `<=`. |
| [ge](./Comparing_Ops/ge.md) | Compares the elements of two tensors, equivalent to `>=`. |
| [lt](./Comparing_Ops/lt.md) | Compares the elements of two tensors, equivalent to `<`. |
| [gt](./Comparing_Ops/gt.md) | Compares the elements of two tensors, equivalent to `>`. |
| [ne](./Comparing_Ops/ne.md) | Compares the elements of two tensors, equivalent to `!=`. |

```{toctree}
:maxdepth: 3
:hidden:

Comparing_Ops/eq.md
Comparing_Ops/le.md
Comparing_Ops/ge.md
Comparing_Ops/lt.md
Comparing_Ops/gt.md
Comparing_Ops/ne.md
```

## Compiler Hint Operations

|api|Description|
|--|--|
|[debug_barrier](./Compiler_Hint_Ops/debug_barrier.md) |Inserts a barrier to synchronize all threads in a block |
|[max_constancy](./Compiler_Hint_Ops/max_constancy.md) |Tells the compiler that the first value in input is constant |
|[max_contiguous](./Compiler_Hint_Ops/max_contiguous.md) |Tells the compiler that the first value in input is contiguous |
|[multiple_of](./Compiler_Hint_Ops/multiple_of.md) |Tells the compiler that all values in input are multiples of value |
|[assume](./Compiler_Hint_Ops/assume.md)         | Provides conditional assumption information to the compiler, allowing it to optimize based on conditions known to be true. |
|[compile_hint](./Extension_Ops/compile_hint.md) | A compiler hint mechanism that allows users to attach metadata to tensors, which is passed to the compiler backend to guide optimization and code generation.|
|[multibuffer](./Extension_Ops/multibuffer.md) | Sets up multi-buffering for a tensor, allowing the compiler to create multiple copies of the same tensor. |
|[parallel](./Extension_Ops/parallel.md) | `parallel` is an iterator specifically designed for multi-core parallel execution, providing explicit multi-core parallelism semantics. |
|[sync_block instruction](./Extension_Ops/sync_block.md) | An explicit inter-core synchronization instruction used to coordinate execution order and data consistency between different cores in the Cube-Vector architecture. |

```{toctree}
:maxdepth: 3
:hidden:

Compiler_Hint_Ops/debug_barrier.md
Compiler_Hint_Ops/max_constancy.md
Compiler_Hint_Ops/max_contiguous.md
Compiler_Hint_Ops/multiple_of.md
Compiler_Hint_Ops/assume.md
Extension_Ops/compile_hint.md
Extension_Ops/multibuffer.md
Extension_Ops/parallel.md
Extension_Ops/sync_block.md
```

## Creation Operations

|api|Description|
|--|--|
|[arange](./Creation_Ops/arange.md) | Returns consecutive values in the half-open interval [start, end) |
|[cat](./Creation_Ops/cat.md) | Concatenates the given blocks |
|[full](./Creation_Ops/full.md) | Returns a tensor filled with a scalar value with the specified shape and dtype|
|[zeros](./Creation_Ops/zeros.md)| Returns a tensor filled with the scalar value 0 with the specified shape and dtype |
|[zeros_like](./Creation_Ops/zeros_like.md)| Returns a zero tensor with the same shape and dtype as the given tensor |
|[cast](./Creation_Ops/cast.md)| Converts a tensor to the specified dtype|

```{toctree}
:maxdepth: 3
:hidden:

Creation_Ops/arange.md
Creation_Ops/cat.md
Creation_Ops/full.md
Creation_Ops/zeros.md
Creation_Ops/zeros_like.md
Creation_Ops/cast.md
```

## Debug Operations

|api|Description|
|--|--|
|[static_print](./Debug_Ops/static_print.md) |Prints values at compile time |
|[static_assert](./Debug_Ops/static_assert.md) |Asserts conditions at compile time |
|[device_print](./Debug_Ops/device_print.md) |Prints values from the device at runtime |
|[device_assert](./Debug_Ops/device_assert.md) |Asserts conditions on the device at runtime |

```{toctree}
:maxdepth: 3
:hidden:

Debug_Ops/static_print.md
Debug_Ops/static_assert.md
Debug_Ops/device_print.md
Debug_Ops/device_assert.md

```

## Indexing and Element Operations

|api|Description|
|--|--|
|[flip](./Indexing_Ops/flip.md) |Flips tensor x along dimension dim |
|[where](./Indexing_Ops/where.md) |Returns a tensor of elements from x or y depending on condition |
|[swizzle2d](./Indexing_Ops/swizzle2d.md) |Converts the index of a row-major matrix of size_i * size_j to the index of a column-major matrix with groups of size_g rows |
|[get_element](./Extension_Ops/get_element.md)| Reads a single element from the input tensor at the given index. |
|[index_select high-performance interface](./Extension_Ops/index_select_simd.md) | Parallelly gathers multiple indices along non-trailing axes and moves data from global memory (GM) directly to the correct positions in the unified buffer (UB) in units of tiles with zero copy. This operation is equivalent to a high-performance implementation of `torch.index_select`, suitable for scenarios such as embedding layer lookups and sparse index access. |
|[gather](./Indexing_Ops/gather.md) | Performs a gather operation on the `src` tensor along the `axis` dimension according to `index` |

```{toctree}
:maxdepth: 3
:hidden:

Indexing_Ops/flip.md
Indexing_Ops/where.md
Indexing_Ops/swizzle2d.md
Extension_Ops/get_element.md
Extension_Ops/index_select_simd.md
Indexing_Ops/gather.md
```

## Inline Assembly

|api|Description|
|--|--|
|[inline_asm_elementwise](./Inline_Assembly/inline_asm_elementwise.md) |Executes inline assembly on tensors |

```{toctree}
:maxdepth: 3
:hidden:

Inline_Assembly/inline_asm_elementwise.md
```

## Iterators

|api|Description|
|--|--|
|[range](./Iterators/range.md)  |An iterator that counts up forever |
|[static_range](./Iterators/static_range.md) | An iterator that counts up forever |

```{toctree}
:maxdepth: 3
:hidden:

Iterators/range.md
Iterators/static_range.md
```

## Linear Algebra Operations

|api|Description|
|--|--|
|[dot](./Linear_Algebra_Ops/dot.md)| Returns the matrix product of two blocks|
|[dot_scaled](./Linear_Algebra_Ops/dot_scaled.md) | Computes the matrix product of two matrix blocks represented in scaled format |

```{toctree}
:maxdepth: 3
:hidden:

Linear_Algebra_Ops/dot.md
Linear_Algebra_Ops/dot_scaled.md
```

## Logical Operations

|api|Description|
|--|--|
|[and](./Logical_Ops/and.md) | Logical AND operation |
|[or](./Logical_Ops/or.md) | Logical OR operation |
|[not](./Logical_Ops/not.md) | Logical NOT operation |
|[logical_and](./Logical_Ops/logical_and.md)| Performs element-wise logical AND on two tensors |
|[logical_or](./Logical_Ops/logical_or.md)| Performs element-wise logical OR on two tensors |
|[invert](./Logical_Ops/invert.md) | Flips each value of the tensor bitwise. |
|[lshift](./Logical_Ops/lshift.md) | Shifts the tensor to the left by the given number of bits. |
|[rshift](./Logical_Ops/rshift.md) | Shifts the tensor to the right by the given number of bits. |
|[xor](./Logical_Ops/xor.md) | Computes the XOR of two elements. |

```{toctree}
:maxdepth: 3
:hidden:

Logical_Ops/and.md
Logical_Ops/or.md
Logical_Ops/not.md
Logical_Ops/logical_and.md
Logical_Ops/logical_or.md
Logical_Ops/not.md
Logical_Ops/invert.md
Logical_Ops/lshift.md
Logical_Ops/rshift.md
Logical_Ops/xor.md
```

## Math Operations

|api|Description|
|--|--|
|[add](./Math_Ops/add.md) | Basic arithmetic addition '+' |
|[sub](./Math_Ops/sub.md) | Basic arithmetic subtraction '-' |
|[mul](./Math_Ops/mul.md) | Basic arithmetic multiplication '*' |
|[div](./Math_Ops/div.md) | Basic arithmetic division '/' |
|[floordiv](./Math_Ops/floordiv.md) | Floor division, basic arithmetic '//' |
|[abs](./Math_Ops/abs.md) |Computes the element-wise absolute value of x |
|[neg](./Math_Ops/neg.md) | Negates the values of the tensor |
|[cdiv](./Math_Ops/cdiv.md) |Computes the ceiling division of x divided by div |
|[ceil](./Math_Ops/ceil.md) |Computes the element-wise ceiling of x |
|[clamp](./Math_Ops/clamp.md) |Clamps the values of the input tensor x to the range [min, max] |
|[cos](./Math_Ops/cos.md) |Computes the element-wise cosine of x |
|[div_rn](./Math_Ops/div_rn.md) |Computes the element-wise exact division of x and y (rounded to the nearest value according to the IEEE standard) |
|[erf](./Math_Ops/erf.md) |Computes the element-wise error function of x |
|[exp](./Math_Ops/exp.md) |Computes the element-wise exponential of x |
|[exp2](./Math_Ops/exp2.md) |Computes the element-wise exponential of x (base 2)|
|[fdiv](./Math_Ops/fdiv.md) |Computes the element-wise fast division of x and y |
|[floor](./Math_Ops/floor.md) |Computes the element-wise floor of x |
|[fma](./Math_Ops/fma.md) |Computes the element-wise fused multiply-add of x, y, and z |
|[log](./Math_Ops/log.md) |Computes the element-wise natural logarithm of x |
|[log2](./Math_Ops/log2.md) |Computes the element-wise logarithm of x (base 2)|
|[mod](./Math_Ops/mod.md) | Modulo operation |
|[maximum](./Math_Ops/maximum.md) |Computes the element-wise maximum of x and y |
|[minimum](./Math_Ops/minimum.md) |Computes the element-wise minimum of x and y |
|[rsqrt](./Math_Ops/rsqrt.md) |Computes the element-wise reciprocal square root of x |
|[sigmoid](./Math_Ops/sigmoid.md) |Computes the element-wise sigmoid of x |
|[sin](./Math_Ops/sin.md) |Computes the element-wise sine of x |
|[softmax](./Math_Ops/softmax.md) |Computes the element-wise softmax of x |
|[sqrt](./Math_Ops/sqrt.md) |Computes the element-wise fast square root of x |
|[sqrt_rn](./Math_Ops/sqrt_rn.md) |Computes the element-wise exact square root of x (rounded to the nearest value according to the IEEE standard) |
|[umulhi](./Math_Ops/umulhi.md)  |Computes the element-wise most significant N bits of the 2N-bit product of x and y |

```{toctree}
:maxdepth: 3
:hidden:

Math_Ops/add.md
Math_Ops/sub.md
Math_Ops/mul.md
Math_Ops/div.md
Math_Ops/floordiv.md
Math_Ops/abs.md
Math_Ops/neg.md
Math_Ops/cdiv.md
Math_Ops/ceil.md
Math_Ops/clamp.md
Math_Ops/cos.md
Math_Ops/div_rn.md
Math_Ops/erf.md
Math_Ops/exp.md
Math_Ops/exp2.md
Math_Ops/fdiv.md
Math_Ops/floor.md
Math_Ops/fma.md
Math_Ops/log.md
Math_Ops/log2.md
Math_Ops/mod.md
Math_Ops/maximum.md
Math_Ops/minimum.md
Math_Ops/rsqrt.md
Math_Ops/sigmoid.md
Math_Ops/sin.md
Math_Ops/softmax.md
Math_Ops/sqrt.md
Math_Ops/sqrt_rn.md
Math_Ops/umulhi.md
```

## Memory/Pointer Operations

|api|Description|
|--|--|
|[load](./Memory_Pointer_Ops/tl.load.md) |Returns a tensor whose values are loaded from the memory location defined by the pointer|
|[store](./Memory_Pointer_Ops/tl.store.md) |Stores the data tensor to the memory location defined by the pointer|
|[make_block_ptr](./Memory_Pointer_Ops/tl.make_block_ptr.md) |Returns a pointer to a block within a parent tensor|
|[advance](./Memory_Pointer_Ops/tl.advance.md) |Advances a block pointer|
|[load_tensor_descriptor](./Memory_Pointer_Ops/load_tensor_descriptor.md) | Loads a data block from a tensor descriptor |
|[make_tensor_descriptor](./Memory_Pointer_Ops/make_tensor_descriptor.md) | Creates a tensor descriptor object |
|[store_tensor_descriptor](./Memory_Pointer_Ops/store_tensor_descriptor.md) | Stores a data block to the memory location specified by the tensor descriptor |

```{toctree}
:maxdepth: 3
:hidden:

Memory_Pointer_Ops/tl.load.md
Memory_Pointer_Ops/tl.store.md
Memory_Pointer_Ops/tl.make_block_ptr.md
Memory_Pointer_Ops/tl.advance.md
Memory_Pointer_Ops/load_tensor_descriptor.md
Memory_Pointer_Ops/make_tensor_descriptor.md
Memory_Pointer_Ops/store_tensor_descriptor.md
```

## Programming Model

|api|Description|
|--|--|
| tensor | An N-dimensional array representing a value or pointer |
| [program_id](./Programming_Model/program_id.md) | Returns the id of the current program instance along the specified axis |
| [num_programs](./Programming_Model/num_programs.md) | Returns the number of program instances along the specified axis |

```{toctree}
:maxdepth: 3
:hidden:

Programming_Model/program_id.md
Programming_Model/num_programs.md
```

## Random Number Generation

|api|Description|
|--|--|
|[randint4x](./Random_Number_Generation/randint4x.md) |Given a seed scalar and an offset block, returns 4 random blocks of type int32 |
|[randint](./Random_Number_Generation/randint.md) |Given a seed scalar and an offset block, returns a random block of type int32 |
|[rand](./Random_Number_Generation/rand.md)   |Given a seed scalar and an offset block, returns a random block of type float32 in U(0,1) |
|[randn](./Random_Number_Generation/randn.md)   |Given a seed scalar and an offset block, returns a random block of type float32 in N(0,1) |

```{toctree}
:maxdepth: 3
:hidden:

Random_Number_Generation/randint4x.md
Random_Number_Generation/randint.md
Random_Number_Generation/rand.md
Random_Number_Generation/randn.md
```

## Reduction Operations

|api|Description|
|--|--|
|[argmax](./Reduction_Ops/argmax.md) |Returns the index of the maximum element in the input tensor along the specified axis |
|[argmin](./Reduction_Ops/argmin.md) |Returns the index of the minimum element in the input tensor along the specified axis |
|[max](./Reduction_Ops/max.md) |Returns the maximum value of all elements in the input tensor along the specified axis |
|[min](./Reduction_Ops/min.md) |Returns the minimum value of all elements in the input tensor along the specified axis |
|[reduce](./Reduction_Ops/reduce.md) |Applies combine_fn to all elements in the input tensor along the specified axis |
|[sum](./Reduction_Ops/sum.md) |Returns the sum of all elements in the input tensor along the specified axis |
|[xor_sum](./Reduction_Ops/xor_sum.md) |Returns the XOR sum of all elements in the input tensor along the specified axis |

```{toctree}
:maxdepth: 3
:hidden:

Reduction_Ops/argmax.md
Reduction_Ops/argmin.md
Reduction_Ops/max.md
Reduction_Ops/min.md
Reduction_Ops/reduce.md
Reduction_Ops/sum.md
Reduction_Ops/xor_sum.md
```

## Scan/Sort Operations

|api|Description|
|--|--|
|[associative_scan](./Scan_Sort_Ops/associative_scan.md) |Applies combine_fn to each element of the input tensor and the carry value along the specified axis, and updates the carry value |
|[cumprod](./Scan_Sort_Ops/cumprod.md) |Returns the cumulative product of all elements in the input tensor along the specified axis |
|[cumsum](./Scan_Sort_Ops/cumsum.md)  |Returns the cumulative sum of all elements in the input tensor along the specified axis |
|[histogram](./Scan_Sort_Ops/histogram.md) |Computes a histogram with num_bins bins of width 1 starting at 0, based on the input tensor |
|[sort](./Scan_Sort_Ops/sort.md) |Sorts the tensor along the specified dimension |

```{toctree}
:maxdepth: 3
:hidden:

Scan_Sort_Ops/associative_scan.md
Scan_Sort_Ops/cumprod.md
Scan_Sort_Ops/cumsum.md
Scan_Sort_Ops/histogram.md
Scan_Sort_Ops/sort.md
```

## Shape Operations

|api|Description|
|--|--|
|[broadcast](./Shape_Manipulation_Ops/broadcast.md) | Attempts to broadcast the two given blocks to a common compatible shape |
|[broadcast_to](./Shape_Manipulation_Ops/broadcast_to.md) | Attempts to broadcast the given tensor to a new shape |
|[expand_dims](./Shape_Manipulation_Ops/expand_dims.md) | Expands the shape of a tensor by inserting new dimensions of length 1
|[interleave](./Shape_Manipulation_Ops/interleave.md) | Interleaves the values of two tensors along the last dimension |
|[join](./Shape_Manipulation_Ops/join.md) | Joins the given tensors in a new minor dimension |
|[permute](./Shape_Manipulation_Ops/permute.md) | Permutes the dimensions of a tensor |
|[ravel](./Shape_Manipulation_Ops/ravel.md) | Returns a contiguous flattened view of x |
|[reshape](./Shape_Manipulation_Ops/reshape.md) | Returns a tensor with the same number of elements as the input but with the provided shape|
|[split](./Shape_Manipulation_Ops/split.md) | Splits a tensor into two parts along its last dimension, which must have size 2 |
|[trans](./Shape_Manipulation_Ops/trans.md) | Transposes the tensor |
|[view](./Shape_Manipulation_Ops/view.md) | Returns a tensor with the same elements as the input but a different shape |
|[extract_slice](./Extension_Ops/extract_slice.md)|  Extracts a tensor from the input tensor according to the offsets, sizes, and strides specified by the operation. |
|[insert_slice](./Extension_Ops/insert_slice.md)| Inserts a tensor (subtensor) into another tensor at the specified position, according to the offsets, sizes, and strides specified by the operation. |

```{toctree}
:maxdepth: 3
:hidden:

Shape_Manipulation_Ops/broadcast.md
Shape_Manipulation_Ops/broadcast_to.md
Shape_Manipulation_Ops/expand_dims.md
Shape_Manipulation_Ops/interleave.md
Shape_Manipulation_Ops/join.md
Shape_Manipulation_Ops/permute.md
Shape_Manipulation_Ops/ravel.md
Shape_Manipulation_Ops/reshape.md
Shape_Manipulation_Ops/split.md
Shape_Manipulation_Ops/trans.md
Shape_Manipulation_Ops/view.md
Extension_Ops/extract_slice.md
Extension_Ops/insert_slice.md
```
