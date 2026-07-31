# triton.language.broadcast

## 1 Function Description

Broadcasts two tensors to a common compatible shape so that they can be used in element-wise operations.

**Syntax:**

- `triton.language.broadcast(input, other)` - function call form

**Functionality:**

- Automatically aligns tensors of different ranks to obtain the target shape
- Expands dimensions of size 1 to the size of the corresponding dimension in the target shape

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | The first input tensor; must be a RankedTensorType |
| other | tensor | Yes | The second input tensor; must be a RankedTensorType |

**Return value:**

- **Type:** tensor
- **Shape:** The common compatible target shape of the two tensors
- **Data type:** Each returned tensor retains the original data type of its input
- **Memory layout:** Returns a newly created tensor

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

**Basic usage:**

```python
@triton.jit
def broadcast_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # Create a scalar (0-dimensional tensor)
    scalar = 5.0

    # Create a vector (1-dimensional tensor)
    vector = tl.arange(0, BLOCK_SIZE) * 1.0  # Shape: (BLOCK_SIZE,)

    # Use broadcast to broadcast the scalar to the same shape as the vector
    # scalar: () -> (BLOCK_SIZE,)
    broadcasted_scalar = tl.broadcast(scalar, vector)

    result = vector + broadcasted_scalar

    # Store the result
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, result)

```
