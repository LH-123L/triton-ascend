# eq

## 1. OP Overview

Description: Used to compare the elements of two tensors element-wise, equivalent to `==`. Used as a built-in operator of `tensor`, e.g., `x==y`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the primary data to be compared |
| `other`   | `tensor` | Tensor data, right operand, compared element-wise with `input` |
| `_builder` | - | Reserved parameter, external invocation not supported for now |

Return value:
`tl.tensor`: A tensor with the same shape as `input`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | √ | √ | × | √ | √ |

Conclusion: Compared with GPU, Triton-Ascend lacks support for uint8/uint16/uint32/uint64 and fp64.

#### 2.2.2 Shape Support

|        | Supported Dimension Range         |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend | Unlimited |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Triton-Ascend lacks fp64 support; uint8/uint16/uint32/uint64 type support is under development.

### 2.4 Usage

The following example performs an `==` operation on tensors `x0` and `x1`:

```python
@triton.jit
def triton_eq(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
        tmp2 = tmp0 == tmp1
        tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)
```
