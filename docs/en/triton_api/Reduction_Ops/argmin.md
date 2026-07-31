# triton.language.argmin

## 1. OP Overview

Description: Returns the index of the minimum value along the specified dimension

```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | Tensor data, left operand |
| `axis`   | `int` | Specifies the dimension on which to reduce |
| `keep_dims` | `bool` | Keeps the reduced dimension after reduction |
| `tie_break_left` | `bool` | If multiple elements have the same minimum value, returns the index of the leftmost minimum value |

Return value:
`tl.tensor`: A tensor containing the index of the minimum value, whose shape depends on the axis and keep_dims parameters; when keep_dims=False, it usually has one fewer dimension than input

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2A3 series| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range         |
| -------- | ---------------------- |
| GPU    | No restrictions |
| Ascend | No restrictions |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Ascend A3 lacks support for uint16, uint32, uint64, and fp64 compared to GPU.

### 2.4 Usage

For more examples, refer to the triton-ascend code repository, ascend/examples/generalization_cases/test_argmin.py

```python
@triton.jit
def triton_argmin_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmin(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```
