# triton.language.abs

## 1. OP Overview

Description: Computes the absolute value of each element in the tensor.
Function Prototype:

```python
triton.language.abs(x, _semantic=None)
```

It can be called as a member function of a tensor, e.g., `x.abs()`, which is equivalent to `abs(x)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `x` | `tensor` | Tensor data |
| `_semantic`   | - | Reserved parameter; not yet supported for external calls |

Return Value:
`out`: A tensor with the same shape as `x`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| -------- | :---: |
| GPU    | No restrictions |
| Ascend | No restrictions |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape.

### 2.3 Special Restrictions

Triton-Ascend does not support fp64 compared with GPU.

### 2.4 Usage Example

The following example performs the absolute value operation on the input tensor `x`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr,
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

    ret = tl.abs(X)

    tl.store(output_ptr + idx, ret)
```
