# triton.language.reshape

## 1 Function Description

Reinterprets a tensor as a new shape.

**Syntax:**

- `triton.language.reshape(input, shape, can_reorder=False)` - function call form
- `input.reshape(shape, can_reorder=False)` - member function form

**Functionality:**

- Reinterprets the tensor as a new shape

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| shape | List[int] | Yes | Target shape |
| can_reorder | bool | No | Whether reordering elements is allowed; defaults to False |

**Return value:**

- **Type:** tensor
- **Shape:** Same as the target shape specified by the shape parameter
- **Data type:** Same as the input tensor
- **Memory layout:** Determined by the can_reorder parameter

**Constraints:**

- The total number of elements of the input and output tensors must be equal
- No dimension smaller than 1 is allowed in any tensor

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Limitations

* The can_reorder parameter only supports False

### 2.5 Usage

```python
import torch
import triton
import triton.language as tl

@triton.jit
def reshape_example(out_ptr):
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Reshape to 6x4
    y = tl.reshape(x, [6, 4])

    # Write the result back to the external tensor
    offs = tl.arange(0, 6)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offs, y)

## Usage example
out = torch.empty((6, 4), dtype=torch.float32, device="npu")
reshape_example[(1,)](out)
print(out.shape)  # Output: torch.Size([6, 4])
```
