# triton.language.make_block_ptr

## 1. OP Overview

Description: Creates a pointer to a tensor on GM
Prototype:

```python
triton.language.make_block_ptr(
 base: triton.PointerType,
 shape: List[tensor],
 strides: tuple(int | constexpr),
 offsets: tuple(int | constexpr),
 block_shape:tuple(int | constexpr),
 order:tuple(constexpr),
 _semantic=None
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                                           |
| ------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `base`        | `triton.PointerType` | The base pointer of the tensor                                                          |
| `shape`       | `tuple(int \| constexpr)` | The shape of the tensor on GM                                                           |
| `strides`     | `tuple(int \| constexpr)` | List of strides for each dimension of the tensor                                        |
| `offsets`     | `tuple(int \| constexpr)` | List of base offsets for each dimension of the tensor                                   |
| `block_shape` | `tuple(constexpr)`  | The shape of the block to be loaded / stored from / to global memory at one time        |
| `order`       | `tuple(constexpr)`  | The shape of the block to be loaded / stored from / to global memory at one time        |
| `_semantic`   | -                   | Reserved parameter, not supported for external calls                                    |

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

#### 2.2.3 Community Constraints

The result of `tl.make_block_ptr` does not allow arithmetic operations. To change the offsets, you can:

1. Re-call make_block_ptr and modify the `offset` parameter:

   ```python
   for block_idx in range(pid, NUM_BLOCKS, 20):
       task_hz_idx = block_idx // NUM_BLOCKS_M
       task_m_idx = block_idx % NUM_BLOCKS_M
       off_z = task_hz_idx // H
       off_h = task_hz_idx % H
       qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
       # Create block pointers for Q, K, V, Output
       Q_block_ptr = tl.make_block_ptr(
           base=Q + qvk_offset,
           shape=(N_CTX, HEAD_DIM),
           strides=(stride_qm, stride_qk),
           offsets=(task_m_idx * BLOCK_M, 0),
           block_shape=(BLOCK_M, HEAD_DIM),
           order=(1, 0),
       )
   ```

2. Call `tl.advance` to adjust the offsets:

   ```python
   block_ptr_in=tl.make_block_ptr(
       base = x_ptr,
       shape = (XB,YB,ZB),
       strides = (YB*ZB,ZB,1),
       offsets = (9,6,5),
       block_shape = (XB,YB,ZB),
       order = (2,1,0),
   )
   bbptr = tl.advance(block_ptr_in,(-9,-6,-5))
   ```

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

- Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

- On Ascend, transpose semantics can only be expressed by adjusting the order of the `order` parameter; transpose semantics cannot be implemented by adjusting the order of the `stride` parameter.

- | Difference                             | Description                                                                                 | Solution                      |
  | -------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------- |
  | Generalization issues when used with branch and loop statements | If `tl.make_block_ptr` is used with complex loop and branch statements, compilation issues may occur | Extensive generalization testing exposes issues; resolved iteratively |

### 2.4 Usage

The following example implements reading a tensor via tl.make_block_ptr and transposing it:

```python
import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common

@pytest.mark.parametrize('shape', [(1, 4, 2)])
@pytest.mark.parametrize('permute_order', [(2, 0, 1)])
def test_makeblockptr_order(shape, permute_order):

    @triton.jit
    def triton_kernel(in0_ptr: tl.tensor, # of tl.pointer_type
                out0_ptr: tl.tensor, # of tl.pointer_type
                in0_stride0: int, in0_stride1: int, in0_stride2: int, # strides for in0
                in0_stride_order0: tl.constexpr, in0_stride_order1: tl.constexpr, in0_stride_order2: tl.constexpr, # stride order for in0
                out0_stride0: int, out0_stride1: int, out0_stride2: int, # strides for out0
                out0_stride_order0: tl.constexpr, out0_stride_order1: tl.constexpr, out0_stride_order2: tl.constexpr, # stride order for out0
                s0: int, s1: int, s2: int,
                tile_size0: tl.constexpr, tile_size1: tl.constexpr, tile_size2: tl.constexpr,
                ):
        tile_id0 = tl.program_id(axis=0)
        tile_id1 = tl.program_id(axis=1)
        tile_id2 = tl.program_id(axis=2)
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        offset2 = (tile_id2 * tile_size2).to(tl.int32)
        in0_bptr = tl.make_block_ptr(in0_ptr,
                                    (s0, s1, s2),
                                    (in0_stride0, in0_stride1, in0_stride2),
                                    (offset0, offset1, offset2),
                                    (tile_size0, tile_size1, tile_size2),
                                    order=(in0_stride_order0, in0_stride_order1, in0_stride_order2))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0, in0_stride_order1, in0_stride_order2)).to(in0_ptr.type.element_ty)

        out0 = in0

        out0_bptr = tl.make_block_ptr(out0_ptr, (s0, s1, s2), (out0_stride0, out0_stride1, out0_stride2), (offset0, offset1, offset2), (tile_size0, tile_size1, tile_size2),
                                    order=(out0_stride_order0, out0_stride_order1, out0_stride_order2))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0, out0_stride_order1, out0_stride_order2))

    def triton_func(in0: torch.Tensor, permute_order):
        # in fact, it adjusts the layout metadata instead of doing a real permutation.
        in0_permuted_tmp = in0.permute(permute_order)
        in0_permuted_shape = in0_permuted_tmp.size()
        in0_permuted_strides = in0_permuted_tmp.stride()
        in0_stride_order = [len(permute_order)-1-i for i in permute_order]
        shape = (in0_permuted_shape[0], in0_permuted_shape[1], in0_permuted_shape[2])
        tile_sizes = (shape[0], shape[1], shape[2])
        out0 = torch.empty(shape, dtype=in0.dtype, device=in0.device)
        out0_strides = out0.stride()
        out0_stride_order = [len(permute_order)-1-i for i in range(len(permute_order))]
        grid = (shape[0]//tile_sizes[0], shape[1]//tile_sizes[1], shape[2]//tile_sizes[2])
        triton_kernel[grid](
                in0, out0,
                in0_permuted_strides[0], in0_permuted_strides[1], in0_permuted_strides[2], # stride for in0
                in0_stride_order[0], in0_stride_order[1], in0_stride_order[2], # stride order for in0
                out0_strides[0], out0_strides[1], out0_strides[2], # stride for out0
                out0_stride_order[0], out0_stride_order[1], out0_stride_order[2], # stride order for out0
                shape[0], shape[1], shape[2], # task indexing space
                tile_size0=tile_sizes[0],
                tile_size1=tile_sizes[1],
                tile_size2=tile_sizes[2],
            )
        return out0

    x0 = torch.randint(0, 9, shape, dtype=torch.int32).npu()
    torch_ref = torch.permute(x0, permute_order)
    triton_cal = triton_func(x0, permute_order)
    test_common.validate_cmp("int32", triton_cal, torch_ref)

```
