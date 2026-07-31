# triton.language.semantic.less_equal

## 1. OP Overview

Description: Used to compare the elements of two tensors element-wise, equivalent to `<=`.
Function prototype:

```python
triton.language.semantic.less_equal(
 input: tl.tensor,
 other: tl.tensor,
 builder: ir.builder
) -> tl.tensor
```

Used as a built-in operator of `tensor`, e.g., `x<=y`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the primary data to be compared |
| `other`   | `tensor` | Tensor data, right operand, compared element-wise with `input` |
| `builder` | - | Reserved parameter, external invocation not supported for now |

Return value:
`tl.tensor`: A tensor with the same shape as `input`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | √ | √ | × | √ | √ |

Conclusion: Compared with GPU, Triton-Ascend does not yet support uint8/uint16/uint32/uint64 and fp64.

#### 2.2.2 Shape Support

|        | Supported Dimension Range         |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend A2/A3| Unlimited |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Triton-Ascend lacks fp64 support; uint8/uint16/uint32/uint64 type support is under development.

### 2.4 Usage

The following example performs a less-than-or-equal-to operation on the three-dimensional tensors `x0` and `x1`:

```python
@triton.jit
def triton_le_3d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 <= x1
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```
