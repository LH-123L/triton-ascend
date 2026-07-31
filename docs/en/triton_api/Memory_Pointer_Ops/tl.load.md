# triton.language.load

## 1. OP Overview

Prototype:

```python
triton.language.load(
 pointer,
 mask=None,
 other=None,
 boundary_check=(),
 padding_option='',
 cache_modifier='',
 eviction_policy='',
 volatile=False,
 _semantic=None
)
```

Description: Returns a Tensor/Scalar whose values are loaded from the location pointed to by the `pointer` parameter in GlobalMemory.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter       | Type                | Description                                                                                                                              |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `pointer`        | `triton.PointerType` <br> or `tensor<triton.PointerType>` <br> or `triton.PointerType<tensor>` (from `tl.make_block_ptr`) | Pointer to the data to be read on GM                                                                                                 |
| `mask`       | `int1` or `tensor<int1>` | Optional parameter; can only be passed when `pointer` does not come from `tl.make_block_ptr`<br>If `mask[i]==False`, the data pointed to by `pointer[i]` will not be read; if `True`, it will be read normally <br>If `pointer` comes from `tl.make_block_ptr`, `mask` must be `None` |
| `other`     | `tensor` or `scalar` | Optional parameter; can only be passed when `mask!=None`<br>If `mask[i]==False`, the i-th position of the return value is set to `other[i]` or `other` (if `other` is a `scalar` type). Tensor support is required because both tensor and scalar are supported in the tritonGPU community |
| `boundary_check` | `tuple(int)` | Optional parameter; can only be passed when `pointer` comes from `tl.make_block_ptr`<br>Tuple of integers indicating the dimensions that require boundary checking |
| `padding_option`   | `""` or `"zero"` or `"nan"` | Optional parameter; can only be passed when `boundary_check` is not empty<br>Indicates the fill value used when accessing out of bounds |
| `cache_modifier`   | `""` or `"ca"` or `"cg"` | Optional parameter; controls the cache option on NVIDIA PTX, has no effect on Ascend hardware |
| `eviction_policy`   | `str`                | Controls the eviction policy of NVIDIA PTX, has no effect on Ascend hardware |
| `volatile`   | `bool`                 | Controls the volatile option of NVIDIA PTX, has no effect on Ascend hardware |
| `_semantic`   | -                 | Reserved parameter, not supported for external calls |

The current 910 generation does not yet support parameters such as cache_modifier, eviction_policy, and volatile

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    |  √    |

Conclusion: Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Supports scalar and 1~5 dimensional tensors |
| Ascend A2/A3 | Supports scalar and 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

#### 2.2.3 Community Constraints

1. If `pointer` is a single pointer:
   - In this case, `tl.load` returns a scalar
   - `mask` and `other` must be scalars
   - `other` is implicitly cast to the data type of `pointer.dtype.element_ty`
   - In this case, `boundary_check` and `padding_option` cannot be passed
2. If `pointer` is an N-Dimensional tensor:
   - In this case, `tl.load` returns an N-Dimensional tensor with the same shape as `pointer`
   - `mask` and `other` are implicitly broadcast to the same shape as `pointer`
   - In this case, `boundary_check` and `padding_option` cannot be passed
3. If `pointer` comes from `tl.make_block_ptr`:
   - In this case, `mask` and `other` must be None
   - In this case, boundary checking and out-of-bounds fill values can be set via `boundary_check` and `padding_option`

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

| Difference                             | Description                                                                                 | Solution                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------- |
| `padding_option` parameter not supported | The community branch currently in use has added the `padding_option` parameter for the out-of-bounds element padding strategy. | Can be supported through software development |
| Generalization issues when used with branch and loop statements | If the computation of `pointer` and `mask` of tl.load involves complex loop and branch statements, compilation issues may occur | Extensive generalization testing exposes issues; resolved iteratively |

### 2.4 Usage

The following example implements the functionality of `torch_ldst_indirect_07_func` through the combined invocation of `triton_ldst_indirect_07_kernel` and `triton_ldst_indirect_07_func`:

```python
@triton.jit
def triton_ldst_indirect_07_kernel(
    out_ptr0, in_ptr0, in_ptr1, in_ptr2, stride_in_r,
    XS: tl.constexpr, RS: tl.constexpr
):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.load(in_ptr0 + in_idx0)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)

def triton_ldst_indirect_07_func(xr, xc, x2, xs, rs):
    nr = x2.size()[0]
    nc = xc.numel()
    stride_in_r = x2.stride()[0]
    assert nr == xs, "test only single core"
    y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
    triton_ldst_indirect_07_kernel[nr // xs, 1, 1](
        y0, xr, xc, x2, stride_in_r, XS = xs, RS = rs)
    return y0

def torch_ldst_indirect_07_func(xr, xc, x2):
    flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
    extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
    return extracted

DEV = "npu"
DTYPE = torch.float32
offset = 8
N0, N1 = 16, 32
blocksize = 4
lowdimsize = N0
assert N1 >= N0+offset, "N1 must be >= N0+offset"
assert N0 == lowdimsize, "N0 must be == lowdimsize"
xc = offset + torch.arange(0, N0, device=DEV)
xr = torch.arange(0, blocksize, device=DEV)
x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
torch_ref = torch_ldst_indirect_07_func(xr, xc, x2)
triton_cal = triton_ldst_indirect_07_func(xr, xc, x2, blocksize, lowdimsize)
torch.testing.assert_close(triton_cal, torch_ref)
```
