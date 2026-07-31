# triton.language.randint

## 1. OP Overview

Description: Given 1 seed scalar and 1 offset block, returns 1 random block of int32 type.
Prototype:

```python
triton.language.randint(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

If multiple random number streams are needed, using randint4x may be faster than calling randint 4 times in succession.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `seed`        | `int` or `tensor`   | The seed used to generate random numbers                                |
| `offset`      | `int` or `tensor`   | The offset used to generate random numbers                              |
| `n_rounds`    | `constexpr`, default is 10 | The number of iteration rounds of the Philox algorithm             |

Return value:
1 random block of int32 type, with the same shape as offset

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

The following example demonstrates the call to randint (generating a single random number per call):

```python
@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds) # Generate a random number for each index
        tl.store(x_ptr + global_offset, rand_vals) # Store the random number

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
```
