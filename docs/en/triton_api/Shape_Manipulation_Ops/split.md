# triton.language.split

## 1 Function Description

Splits the input tensor into two tensors along the last dimension. The last dimension of each output tensor is half the size of the input tensor's last dimension, and the other dimensions remain unchanged.

**Syntax:**

- `triton.language.split(input)` - function call form
- `input.split()` - member function form

**Functionality:**

- Splits the input tensor into two tensors along the last dimension
- The last dimension of the output tensors is half that of the input tensor; the size of the last dimension must be 2
- The other dimensions remain unchanged

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |

**Return value:**

- **Type:** Tuple[tensor, tensor]
- **Shape:** Two tensors with the same shape, whose last dimension is half that of the input
- **Data type:** Same as the input tensor
- **Memory layout:** Contain the elements at the odd and even positions of the input tensor respectively

**Constraints:**

- The size of the last dimension of the input tensor must be even
- Outputs two tensors with the same shape

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
@triton.jit
def complex_split_kernel(complex_ptr, real_ptr, imag_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Load the complex data
    complex_data = tl.load(complex_ptr + offsets, mask=mask)

    # Split into real and imaginary parts
    real_part, imag_part = complex_data.split()

    # Store the real and imaginary parts
    tl.store(real_ptr + offsets, real_part, mask=mask)
    tl.store(imag_ptr + offsets, imag_part, mask=mask)
```
