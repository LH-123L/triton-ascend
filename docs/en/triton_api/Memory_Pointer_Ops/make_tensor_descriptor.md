# triton.language.make_tensor_descriptor

## 1. OP Overview

Description: Create a tensor descriptor object
Prototype (Triton 3.4.0 version):

```python
triton.language.make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _semantic=None
) -> tensor_descriptor
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                                                  |
| ------------- | ------------------- | ------------------------------------------------------------------------------------------- |
| `base`        | `tensor`            | The base pointer of the tensor                                                               |
| `shape`       | `List[tensor]`      | The shape of the tensor                                                                      |
| `strides`     | `List[tensor]`      | List of strides for each dimension of the tensor, with the following constraints: - The leading dimensions must be integer multiples of 16 bytes - The last dimension must be stored contiguously |
| `block_shape` | `List[constexpr]`   | The shape of the block to be loaded / stored from / to global memory                         |
| `_semantic`   | -                   | Reserved parameter, not supported for external calls                                         |

Return value:
`tensor_descriptor`: Tensor descriptor object (cannot be used in arithmetic operations directly; it must be used with `load` / `store`)

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | × |

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).

| Difference                 | Description                                                                        | Solution                                              |
| -------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Binding usage restriction  | `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` must be used together and cannot be mixed with `tl.load()` / `tl.store()`. | Upgrading to Triton 3.4.0 to sync upstream functions (such as `cast`) can resolve this |
| `padding_option` parameter not supported | The current community mainline branch has added the `padding_option` parameter for the out-of-bounds element padding strategy. | Can be supported through software development        |
| Triton version compatibility | Triton 3.2.0 has compatibility issues with some functions (such as `cast`). It is recommended to upgrade the Triton version to 3.4.0 to fix the binding restriction. | Upgrade to Triton 3.4.0                               |

### 2.4 Usage

The following example computes the in-place absolute value of the input tensor `x`:

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # Create a tensor descriptor
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
 # Calculate the offset corresponding to the current thread
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
 # Load the data, compute the absolute value, and store the result
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
## Initialize the tensor
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## Configure the block size and grid
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
