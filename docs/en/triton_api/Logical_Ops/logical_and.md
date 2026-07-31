# triton.language.tensor.logical_and

## 1. OP Overview

Description: Used to perform element-wise logical AND on two tensors

```python
x.logical_and(y)
```

Called as a member function of `tensor`, e.g., `x0.logical_and(x1)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the primary data to be compared |
| `other`   | `tensor` | Tensor data, right operand, performs element-wise logical AND with `input` |
| `_builder` | - | Reserved parameter, external invocation not supported for now |

Return value:
`tl.tensor`: A tensor with the same shape as `input`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | × | × | × | × | × | × | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

Conclusion: In terms of DataType, compared with GPU, Ascend additionally supports integer and floating-point types (except fp64 and fp8).

#### 2.2.2 Shape Support

|        | Supported Dimension Range         |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend A2/A3 | Unlimited |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability not yet supported relative to the community

None.

### 2.4 Usage

The following example performs a logical AND operation on the three-dimensional tensors `x0` and `x1`:

```python
@triton.jit
def triton_logical_and_3d(in_ptr0, in_ptr1, out_ptr0, XB, YB, ZB, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L) + tl.program_id(0) * XB
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * YB
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * ZB
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_and(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```
