# triton.language.advance

## 1. OP Overview

Description: Adds an offset to the offset of `tl.make_block_ptr`
Prototype:

```python
triton.language.advance(
 base: triton.PointerType,
 offsets: tuple(int | constexpr),
 _semantic=None
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                                           |
| ------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `base`        | `triton.PointerType` | The pointer to be updated, the result of `tl.make_block_ptr`                           |
| `offsets`     | `tuple(int \| constexpr)` | List of base offsets for each dimension of the tensor; `len(offsets)` must be equal to `len(base.offsets)` |
| `_semantic`   | -                   | Reserved parameter, not supported for external calls                                  |

Return value: `pointer_type<blocked<shape, element_type>>`: Pointer to the tensor

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    | ×    |

Conclusion: Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

- Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).
- On Ascend, transpose semantics can only be expressed by adjusting the order of the `order` parameter; transpose semantics cannot be implemented by adjusting the order of the `stride` parameter.
- Currently, `tl.make_tensor_ptr` may cause compilation issues when used with complex loop and branch statements

### 2.4 Usage

Refer to the following example:

```python
@triton.jit
def fn_npu_3d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(3, 1, 2),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-3, -1, -2))
    # XB,YB,1
    X = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)
```
