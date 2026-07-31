# triton.language.topk

## 1. Function Overview

Description: Returns the top `k` largest elements of the input tensor `x` along the specified dimension, with the results sorted in descending order.

```python
triton.language.topk(x, k, dim: constexpr | None = None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| ------ | ---- | ---- |
| `x` | `tensor` | Input tensor |
| `k` | `int` | The number of top elements to return; must be a power of 2 |
| `dim` | `constexpr int` or `None` | The dimension in which to find the top k elements; this parameter must be determined at compile time; if `None`, the last dimension is used; currently only the last dimension is supported |

Return value:
`out`: The output tensor has the same shape as the input tensor, except that the length of the specified dimension becomes `k`

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | ×      | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √ | √ | × | × | × | × | × | × | √ | √ | × | √ | × |

Conclusion: Ascend lacks support for int32, uint8, int64, float64, and bool compared to GPU.
torch_npu supports uint8.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | ------------ |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Due to Bisheng compiler limitations, int32, uint8, int64, float64, and bool cannot be implemented.

Currently, `topk` only returns the maximum values; it does not support switching to returning minimum values via a parameter.
`dim` only supports the last dimension.
`k` must be a power of 2.

### 2.4 Usage

The following example takes the top `k` largest elements of the input tensor `x` along the last dimension:

```python
@triton.jit
def topk_kernel_2d(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid
    offs_n = tl.arange(0, N)
    offs = offs_m * N + offs_n
    x = tl.load(X + offs)
    z = tl.topk(x, K, dim=0)
    tl.store(Z + offs_m * K + tl.arange(0, K), z)
```
