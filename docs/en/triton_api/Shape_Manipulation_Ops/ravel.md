# triton.language.ravel

## 1 Function Description

Flattens the input tensor into a 1-dimensional tensor, preserving the order of elements in memory. The total number of elements in the output tensor is the same as that of the input tensor.

**Syntax:**

- `triton.language.ravel(input)` - function call form
- `input.ravel()` - member function form

**Functionality:**

- Flattens the input tensor into a 1-dimensional tensor
- Preserves the order of elements in memory
- The total number of elements in the output tensor is the same as that of the input tensor

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |

**Return value:**

- **Type:** tensor
- **Shape:** A 1-dimensional tensor containing all elements of the input tensor
- **Data type:** Same as the input tensor
- **Memory layout:** Flattened in row-major order

**Constraints:**

- No special constraints; inputs of any shape are supported

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
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    # Load the 2D data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Flatten to 1 dimension
    x_flat = x.ravel()

    # Store the flattened result
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```
