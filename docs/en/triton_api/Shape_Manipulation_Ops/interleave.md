# triton.language.interleave

## 1 Function Description

Interleaves two input tensors of the same shape along the last dimension. The last dimension of the output tensor is twice the size of the input tensors, and the other dimensions remain unchanged.

**Syntax:**

- `triton.language.interleave(x, y)` - function call form
- `x.interleave(y)` - member function form

**Functionality:**

- Interleaves two input tensors of the same shape along the last dimension
- The last dimension of the output tensor is twice the size of the input tensors
- The other dimensions remain unchanged

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| x | tensor | Yes | The first input tensor |
| y | tensor | Yes | The second input tensor; its shape must be the same as x |

**Return value:**

- **Type:** tensor
- **Shape:** The last dimension of the input shape multiplied by 2
- **Data type:** Same as the input tensors
- **Memory layout:** The elements of x and y are interleaved

**Constraints:**

- The two input tensors must have the same shape and data type
- The shape of the output tensor is the input shape with the last dimension multiplied by 2

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
import triton
import triton.language as tl

@triton.jit
def interleave_example():
    # Create two 2x3 tensors
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)

    # Interleave to make it 2x6
    z = tl.interleave(x, y)

    return z

## Usage example
result = interleave_example()
print(result.shape)  # Output: (2, 6)
```
