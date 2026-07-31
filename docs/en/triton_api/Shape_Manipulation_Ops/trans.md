# triton.language.trans

## 1 Function Description

Transposes the dimensions of a tensor according to the dims parameter without changing the tensor's data, only the order of the dimensions. It is a specially optimized transpose operation.

**Syntax:**

- `triton.language.trans(input, dims)` - function call form
- `input.trans(dims)` - member function form

**Functionality:**

- Transposes the dimensions of the tensor according to the dims parameter
- Does not change the tensor's data, only the order of the dimensions
- A specially optimized transpose operation

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | The dimension order after transposition |

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
import triton
import triton.language as tl

@triton.jit
def trans_example():
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Transpose the dimensions to make it 4x2x3
    y = tl.trans(x, [2, 0, 1])

    return y

## Usage example
result = trans_example()
print(result.shape)  # Output: (4, 2, 3)
```
