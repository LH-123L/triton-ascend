# div

## 1. OP Overview

Description: Division, the '/' arithmetic operation; there is no tl.div method

The underlying implementation is the same as the fdiv operator, except that fdiv explicitly restricts the input arguments to float types, while '/' has no such restriction: it converts non-float types to float types before computing

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `tensor or Number`     |     First input, the dividend    |
| `other`       | `tensor or Number`     |     Second input, the divisor    |

Return Value:
`tl.tensor`: The division result
Return result type: Always returns a float type

| Input Type            | Handling                 | Result Type      |
| --------------------- | -------------------------- | --------------- |
| `int / int`     | Both are converted to `float32` | `float32` |
| `int / float`   | int is converted to float             | float type    |
| `float / float` | Unified to the higher-precision float     | Higher-precision float  |
| `float / int`   | int is converted to float             | float type    |

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| × | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | No restrictions |
| Ascend A2/A3 | No restrictions  |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape.

### 2.3 Special Restrictions

Ascend A3 lacks support for uint8, uint16, uint32, uint64, and fp64 compared with GPU.

### 2.4 Usage Example

The following example performs the division operation on the input tensors `in_ptr0, in_ptr1`:

```python
@triton.jit
def triton_div(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 / tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```
