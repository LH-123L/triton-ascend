# triton.language.expand_dims

## 1 Function Description

Inserts a dimension of size 1 at the specified axis position without changing the tensor's data, only increasing the number of dimensions. Negative indexing is supported, counting from right to left.

**Syntax:**

- `triton.language.expand_dims(input, axis)` - function call form
- `input.expand_dims(axis)` - member function form

**Functionality:**

- Inserts a dimension of size 1 at the specified axis position
- Does not change the tensor's data, only increasing the number of dimensions
- Supports negative indexing, counting from right to left

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| axis | int \| Tuple[int] | Yes | The position where the dimension is inserted; negative indexing is supported |

**Return value:**

- **Type:** tensor
- **Shape:** A dimension of size 1 is inserted at the specified axis position
- **Data type:** Same as the input tensor
- **Memory layout:** Implemented via tensor::ExpandShapeOp; no data copy

**Constraints:**

- axis must be in the range [-rank-1, rank], where rank is the number of dimensions of the input tensor
- The inserted dimension size is fixed to 1

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Limitations

None

### 2.5 Usage

```python
import torch
import triton
import triton.language as tl

@triton.jit
def expand_dims_example(out_ptr):
    # Create a 2x3 tensor
    x = tl.zeros([2, 3], dtype=tl.float32)

    # Insert a dimension at axis=1, making it 2x1x3
    y = tl.expand_dims(x, axis=1)

    # Write the result back to the external tensor
    offs = (
        tl.arange(0, 2)[:, None, None] * 3
        + tl.arange(0, 1)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## Usage example
out = torch.empty((2, 1, 3), dtype=torch.float32, device="npu")
expand_dims_example[(1,)](out)
print(out.shape)  # Output: torch.Size([2, 1, 3])
```
