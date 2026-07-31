# triton.language.load_tensor_descriptor

## 1. OP Overview

Description: This function is used to load a data block from a tensor descriptor.

```python
triton.language.load_tensor_descriptor(
    desc: tensor_descriptor_base,
    offsets: Sequence[constexpr | tensor],
    _semantic=None
) -> tensor
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter   | Type                            | Description                                                                      |
| ----------- | ------------------------------- | ------------------------------------------------------------------------------- |
| `desc`      | `tensor_descriptor_base`        | Tensor descriptor object created by `make_tensor_descriptor`, which defines the memory layout (shape, strides, block size, etc.). |
| `offsets`   | `Sequence[constexpr \| tensor]` | Sequence of starting offsets for data loading, used to specify the data location to be loaded by the current thread block |
| `_semantic` | -                               | Reserved parameter, not supported for external calls                             |

Return value: `tensor` - The data block loaded from the specified offsets according to the memory layout information of the tensor descriptor

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | × |

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).

| Difference            | Description                                                                        | Solution                                                    |
| --------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Binding usage restriction | `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` must be used together and cannot be mixed with `tl.load()` / `tl.store()`. | Upgrading to Triton 3.4.0 to sync upstream functions (such as `cast`) can resolve this |
| Triton version compatibility | Triton 3.2.0 has compatibility issues with some functions (such as `cast`). It is recommended to upgrade the Triton version to 3.4.0 to fix the binding restriction. | Upgrade to Triton 3.4.0                                     |

### 2.4 Usage

`load_tensor_descriptor` provides two calling forms:

* Object-oriented method call (recommended)

```python
value = desc.load(offsets)
```

* Functional interface call

```python
value = triton.language.load_tensor_descriptor(desc, offsets)
```

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
import torch
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## Configure the block size and grid
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
