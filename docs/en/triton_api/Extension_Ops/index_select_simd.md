# triton.language.extra.ascend.libdevice.index_select_simd

## 1 Function Description

Parallelly gathers multiple indices along non-trailing-axis dimensions and zero-copies data directly from global memory (GM) to the correct position in the unified buffer (UB) in tiles. This operation is equivalent to a high-performance implementation of `torch.index_select` and is suitable for scenarios such as embedding layer lookups and sparse index access.

**Syntax:**

- `triton.language.extra.ascend.libdevice.index_select_simd(src, dim, index, src_shape, src_offset, read_shape)`

**Function:**

- Reads data in batches from the specified dimension of the source tensor based on the index array
- Supports specifying the offset and size of the read region for flexible slicing
- Zero-copy high-efficiency implementation that moves data directly from GM to UB
- Preserves the element type and encoding unchanged

**Typical application scenarios:**

- Embedding layer lookup: reads word vectors in batches from a large vocabulary based on token IDs
- Sparse tensor operations: accesses specific rows of a dense tensor based on sparse indices
- Dynamic routing and attention mechanisms: selects specific features based on dynamically computed indices

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| src | tensor/pointer | Yes | Source tensor pointer, data located in global memory (GM) |
| dim | int | Yes | The dimension on which the index_select operation is performed, in the range [0, len(src_shape)-2]; **the trailing axis (last dimension) is not supported** |
| index | tensor | Yes | 1D index array located in UB, specifying the index positions to read |
| src_shape | Tuple[int] | Yes | The full shape of the source tensor |
| src_offset | Tuple[int] | Yes | The position from which to start reading; can be set to -1 for the dim dimension (that dimension is determined by index) |
| read_shape | Tuple[int] | Yes | The size of the data to read; must be set to -1 for the dim dimension (that dimension is determined by the length of index) |

**Return value:**

- **Type:** tensor (located in UB)
- **Shape:** Same as read_shape, where the size of the dim dimension equals the length of index
- **Data type:** Same as the source tensor
- **Memory location:** Unified buffer (UB)

**Constraints:**

- `read_shape[dim]` must be -1
- `src_offset[dim]` can be set to -1 (it will be ignored because that dimension is determined by index)
- `len(src_shape) == len(src_offset) == len(read_shape)`
- `index` must be a 1D tensor
- `dim` cannot be the trailing axis (last dimension), i.e., `dim < len(src_shape) - 1`
- For non-dim dimensions: `0 <= src_offset[i] < src_shape[i]`
- For non-dim dimensions: `src_offset[i] + read_shape[i] <= src_shape[i]` (out-of-bound values are automatically truncated)
- The index values in index must be within the range `[0, src_shape[dim])`

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Note:**

- The data type of index must be int32 or int64
- This operation is not supported on the GPU platform (Ascend-specific intrinsic)

### 2.3 Shape Support Table

Any number of dimensions is supported (from 1D to high-dimensional tensors), provided that the following conditions are met:

- index must be a 1D tensor
- The size of each dimension of the source tensor must comply with actual hardware memory limits
- The sizes of read_shape on non-dim dimensions must consider UB space limits

**Common shape combinations:**

- 2D tensors: suitable for embedding layer lookups and sparse matrix row selection
- 3D tensors: suitable for batched embedding lookups and sequence feature extraction
- High-dimensional tensors: suitable for complex multi-dimensional indexing operations

### 2.4 Special Limitations

1. **dim limitation:** index_select cannot be performed on the trailing axis (last dimension); dim must satisfy `dim < len(src_shape) - 1`
2. **Data type limitation:** uint16/uint32/uint64/float8/float64 data types are not supported for now
3. **Index out of bounds:** Whether the indices in index are out of bounds is not checked; users must ensure the validity of the indices themselves

### 2.5 Usage

**Basic usage (2D embedding lookup):**

```python
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def embedding_kernel(
    embed_ptr,      # [vocab_size, embed_dim]
    indices_ptr,    # [batch_size]
    output_ptr,     # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load indices
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))

    # Use index_select to read embedding vectors in batch
    embeddings = libdevice.index_select_simd(
        src=embed_ptr,
        dim=0,
        index=indices,
        src_shape=(vocab_size, embed_dim),
        src_offset=(-1, 0),
        read_shape=(-1, embed_dim)
    )

    # Store the result
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
```

**Relationship with torch.index_select:**

- `index_select_simd` is equivalent to `torch.index_select(src, dim, index)` plus slicing
- However, index_select_simd is implemented at the hardware level, with better performance than the PyTorch implementation (approximately 0.6~1.5x the performance of AscendC)

**Differences from a regular load:**

```python
## Regular load approach (inefficient)
for i in range(len(indices)):
    idx = tl.load(indices_ptr + i)
    offsets = idx * stride + tl.arange(0, size)
    data = tl.load(src_ptr + offsets)
    # ... process data

## index_select approach (efficient)
indices = tl.load(indices_ptr + tl.arange(0, len(indices)))
data = libdevice.index_select_simd(
    src=src_ptr,
    dim=0,
    index=indices,
    src_shape=(...),
    src_offset=(-1, 0),
    read_shape=(-1, size)
)
## Get all the data at once
```

## 3 Differences from GPU

New OP, no differences

## 4 Test Case Description

**Test file:**

- `ascend/examples/pytest_ut/test_index_select.py` - 2D tensor index_select test (multiple shape combinations)
