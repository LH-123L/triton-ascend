# triton.language.randn

## 1. OP Overview

Description: Given 1 seed scalar and 1 offset block, returns 1 random block of float32 type in **N**(**0**,**1**) (following the standard normal distribution).
Prototype:

```python
triton.language.randn(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `seed`        | `int` or `tensor`   | The seed used to generate random numbers                                |
| `offset`      | `int` or `tensor`   | The offset used to generate random numbers                              |
| `n_rounds`    | `constexpr`, default is 10 | The number of iteration rounds of the Philox algorithm             |

Return value:
1 random block of float32 type, with the same shape as offset, whose values follow the standard normal distribution `N(0, 1)`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

Type of the input seed:

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

### 2.4 Usage

The following example demonstrates the call to randn:

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    offsets = block_offset + tl.arange(0, XBLOCK)  # Block-level offset tensor
    mask = offsets < N
    rand_vals = tl.randn(5, 10 + offsets, n_rounds)  # Generate an entire block of random numbers at once
    tl.store(x_ptr + offsets, rand_vals, mask=mask)

shape = (1024,)
y_calf = torch.zeros(shape, dtype=torch.float32).npu()
numel = y_calf.numel()
ncore = 1 if numel < 32 else 32
xblock = math.ceil(numel / ncore)
kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)
```
