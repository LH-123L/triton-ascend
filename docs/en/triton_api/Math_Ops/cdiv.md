# triton.language.cdiv

## 1. OP Overview

Description: Computes the ceiling division of a tensor
Function Prototype:

```python
triton.language.cdiv(x, div)
```

It can be called as a member function of a tensor, e.g., `x.cdiv(...)`, which is equivalent to `cdiv(x, ...)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `x` | `tensor` | Tensor data, the dividend |
| `div`   | `tensor` | Tensor data, the divisor |

Return Value:
`out`: A tensor with the same shape as `x` and `div`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | × | × | × | × | × |

Conclusion: Compared with GPU, Ascend does not support uint or bool inputs.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | No restrictions |
| Ascend | No restrictions |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape.

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

Input range: 0~16777216

### 2.4 Usage Example

The following example performs the ceiling division operation on the input tensors `x` and `y`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cdiv(X, Y)

    tl.store(output_ptr + idx, ret)
```
