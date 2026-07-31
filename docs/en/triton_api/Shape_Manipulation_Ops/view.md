# triton.language.view

## 1 Function Description

Creates a view of a tensor, changing its shape without copying data. It is similar to reshape, but places more emphasis on the concept of a view, keeping the data contiguous in memory.

**Syntax:**

- `triton.language.view(input, shape)` - function call form
- `input.view(shape)` - member function form

**Functionality:**

- Creates a view of the tensor, changing its shape without copying data
- Similar to reshape, but places more emphasis on the concept of a view
- Keeps the data contiguous in memory

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| shape | List[int] | Yes | Target shape |

**Return value:**

- **Type:** tensor
- **Shape:** Same as the target shape specified by the shape parameter
- **Data type:** Same as the input tensor
- **Memory layout:** Contiguous with the input tensor in memory

**Constraints:**

- The total number of elements of the input and output tensors must be equal
- The output tensor must be contiguous with the input tensor in memory

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
def view_example(out_ptr):
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Create a view to make it 6x4
    y = tl.view(x, [6, 4])

    # Write the result back to the external tensor
    offs = tl.arange(0, 6)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offs, y)

## Usage example
out = torch.empty((6, 4), dtype=torch.float32, device="npu")
view_example[(1,)](out)
print(out.shape)  # Output: torch.Size([6, 4])
```
