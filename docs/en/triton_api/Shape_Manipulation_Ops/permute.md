# triton.language.permute

## 1 Function Description

Rearranges the dimensions of a tensor according to the dims parameter without changing the tensor's data, only changing the order of the dimensions. Rearrangement of any number of dimensions is supported.

**Syntax:**

- `triton.language.permute(input, dims)` - function call form
- `input.permute(dims)` - member function form

**Functionality:**

- Rearranges the dimensions of the tensor according to the dims parameter
- Does not change the tensor's data, only the order of the dimensions
- Supports rearrangement of any number of dimensions

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | The new dimension order |

**Return value:**

- **Type:** tensor
- **Shape:** Dimensions rearranged according to the dims parameter
- **Data type:** Same as the input tensor
- **Memory layout:** Transpose is implemented by changing stride information; no data copy

**Constraints:**

- dims must contain all dimension indices of the input tensor

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Limitations

* Transposes with more than 8 dimensions are not supported

### 2.5 Usage

```python
import torch
import triton
import triton.language as tl

@triton.jit
def permute_example(out_ptr):
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Transpose the dimensions to make it 4x2x3
    y = tl.permute(x, [2, 0, 1])

    # Write the result back to the external tensor
    offs = (
        tl.arange(0, 4)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## Usage example
out = torch.empty((4, 2, 3), dtype=torch.float32, device="npu")
permute_example[(1,)](out)
print(out.shape)  # Output: torch.Size([4, 2, 3])
```
