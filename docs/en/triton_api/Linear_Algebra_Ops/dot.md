# triton.language.dot

## 1. OP Overview

Description: Performs matrix multiplication on two tensors. The tensors need to be two-dimensional or three-dimensional with consistent dimensions. For three-dimensional blocks, tl.dot performs batched matrix multiplication, where the first dimension of each block represents the batch dimension.
Prototype:

```python
triton.language.dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=triton.language.float32, _semantic=None)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input`        | `int8 fp16 bf16 fp32`     |     The first input, a 2D or 3D tensor. The value range is limited to -5 to 5 to avoid overflow     |
| `other`       | `int8 fp16 bf16 fp32`     |     The second input, a 2D or 3D tensor. The value range is limited to -5 to 5 to avoid overflow    |
| `acc`           | `int32  float32`    | A tensor storing the accumulation result, accumulator tensor. If not None, the result is added to this tensor. acc_dtype supports {:code:`float16`, :code:`float32`, :code:`int32`} |
| `input_precision`   | -                 |  NVIDIA decides whether to enable Tensor Cores acceleration by selecting a precision mode    |
| `max_num_imprecise_acc`     | `int`    | The number of low-precision accumulations (the current Ascend does not support low-precision accumulation) |
| `out_dtype`     | `fp32  int32`    | The output result type|

Return value:
`tl.tensor`: The matrix multiplication result

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|   Input Type     | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ✓    |

Conclusion: Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Unlimited |
| Ascend A2/A3 |Unlimited  |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

- Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

- acc does not support fp16; the hardware defaults to fp32 for precision

- max_num_imprecise_acc is not supported for now

- Compared with GPU, out_dtype does not support int8 and fp16 types

### 2.4 Usage

The following example performs matrix multiplication on the input tensors `x_ptr, y_ptr`; refer to ascend/examples/generalization_cases/test_matmul.py:

```@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        acc_dtype: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```
