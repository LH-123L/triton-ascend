# triton.language.dot_scaled

## 1. OP Overview

Description: **Computes the matrix product of two matrix blocks represented in scaled format**

```python
triton.language.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
    acc=None, lhs_k_pack=True, rhs_k_pack=True,
    out_dtype=triton.language.float32, _semantic=None)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `lhs`        | `tensor`          | The base pointer of the left matrix tensor (supports bf16 and fp16 formats)                                                      |
| `lhs_scale`        | `tensor`          | The base pointer of the left matrix scale tensor (supports int8 format)                                                      |
| `lhs_format`        | `string`          | The storage format of the left matrix tensor (supports "bf16" and "fp16")                                                     |
| `rhs`        | `tensor`          | The base pointer of the right matrix tensor (supports bf16 and fp16 formats)                                                     |
| `rhs_scale`        | `tensor`          | The base pointer of the right matrix scale tensor (supports int8 format)                                                       |
| `rhs_format`        | `string`          | The storage format of the right matrix tensor (supports "bf16" and "fp16")                                                      |
| `acc`       | `tensor`    | The accumulation tensor                                                        |
| `lhs_k_pack`     | `(bool, optional)`    | true: pack along the K dimension<br>false: pack along the M dimension<br>|
| `rhs_k_pack` | `(bool, optional)` | true: pack along the K dimension<br>false: pack along the N dimension<br>|
| `_semantic`   | -                 | Reserved parameter, external invocation not supported for now                                                |

Return value:
`out`: A tensor, the value output after computing the scaled matrix multiplication

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        |     fp4     |    fp8    |    bf16    |    fp16    |
| ------------- | --------- | -------- | -------- | -------- |
| GPU    | √    | √     | √     | √     |
| Ascend A2/A3 | ×    | ×     | √     | √    |

Conclusion:
1. Compared with GPU, Ascend lacks support for fp4 and fp8 (hardware limitation).
2. The values of the scale tensor are int8; on GPU they are uint8.

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | 2~3-dimensional tensors supported |
| Ascend | 2~3-dimensional tensors supported |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; the lhs/rhs matrices both support 2- to 3-dimensional tensors, but the scale matrix only supports 2 dimensions.

### 2.3 Special Limitations

1. Since fp8 is not supported, the left and right matrices do not support fp4/fp8 formats; compared with GPU, Ascend lacks support for lhs_k_pack/rhs_k_pack matrix unpacking (hardware limitation).
2. The recommended input range for the input matrices lhs and rhs is [-5, 5]; exceeding this range may produce the extreme value inf.
3. Due to hardware alignment requirements, the multiple by which the scale matrix is broadcast must be constrained and should be at least 16.

4. The currently supported scale matrix format is int8, while the community uses uint8

### 2.4 Usage

The following example performs an in-place absolute value computation on the input tensor `x`:

```python
@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                     type_b: tl.constexpr):
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
    str_a0: tl.constexpr = stride_a0
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                            str_a0)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                     BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)

    out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, accumulator.to(a.dtype))

x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
y = torch.randn(shape, dtype=torch.bfloat16, device="npu")
M, K = shape[0], shape[1]
scale_x = torch.randint(min_scale - 128, max_scale - 127, (M, K // 32), dtype=torch.int8, device="npu")
scale_y = torch.randint(min_scale - 128, max_scale - 127, (N, K // 32), dtype=torch.int8, device="npu")
type_a, type_b = "bf16", "bf16"
pgm = dot_scale_kernel[(1,)](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b)
```
