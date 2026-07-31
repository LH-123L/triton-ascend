# triton.language.store_tensor_descriptor

## 1. OP Overview

Description: Stores a data block to the memory location specified by the tensor descriptor

```python
triton.language.store_tensor_descriptor(
    desc: tensor_descriptor_base,
    offsets: Sequence[constexpr | tensor],
    value: tensor,
    _semantic=None
) -> tensor
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter   | Type                            | Description                                                                      |
| ----------- | ------------------------------- | ------------------------------------------------------------------------------- |
| `desc`      | `tensor_descriptor_base`        | Tensor descriptor object created by `make_tensor_descriptor`, which defines the memory layout (shape, strides, block size, etc.). |
| `offsets`   | `Sequence[constexpr \| tensor]` | Sequence of starting offsets for data storage, used to specify the data location to be stored by the current thread block |
| `value`     | `tensor`                        | The tensor data block to be written                                              |
| `_semantic` | -                               | Reserved parameter, not supported for external calls                             |

Return value: `tensor` - The data block actually written

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
| Ascend | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).

### 2.4 Usage

`store_tensor_descriptor` provides two calling forms:

* Object-oriented method call (recommended)

```python
desc.store(offsets, value)
```

* Functional interface call

```python
triton.language.store_tensor_descriptor(desc, offsets, value)
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
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## Configure the block size and grid
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
