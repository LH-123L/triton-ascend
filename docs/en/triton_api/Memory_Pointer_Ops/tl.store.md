# triton.language.store

## 1. OP Overview

Prototype:

```python
triton.language.store(
 pointer,
 value,
 mask=None,
 boundary_check=(),
 cache_modifier='',
 eviction_policy='',
 _semantic=None
)
```

Description: Stores a Tensor/Scalar back to GlobalMemory at the address pointed to by `pointer`

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter       | Type                | Description                                                                                                                              |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `pointer`        | `triton.PointerType` <br> or `tensor<triton.PointerType>` <br> or `triton.PointerType<tensor>` (from `tl.make_block_ptr`) | Pointer to the address to be stored on GM                                                                                            |
| `value`       | `tensor` or `scalar` | The value to be stored; implicit broadcasting and implicit type conversion are supported |
| `mask`       | `int1` or `tensor<int1>` | Optional parameter; can only be passed when `pointer` does not come from `tl.make_block_ptr`<br>If `mask[i]==False`, `value[i]` will not be stored to the address pointed to by `pointer[i]`; if `True`, it will be stored normally <br>If `pointer` comes from `tl.make_block_ptr`, `mask` must be `None` |
| `boundary_check` | `tuple(int)` | Optional parameter; can only be passed when `pointer` comes from `tl.make_block_ptr`<br>Tuple of integers indicating the dimensions that require boundary checking |
| `cache_modifier`   | `""` or `"ca"` or `"cg"` | Optional parameter; controls the cache option on NVIDIA PTX, has no effect on Ascend hardware |
| `eviction_policy`   | `str`                | Controls the eviction policy of NVIDIA PTX, has no effect on Ascend hardware |
| `_semantic`   | -                 | Reserved parameter, not supported for external calls |

Return value: No return value

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    |  √    |

Conclusion: Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).
Expert note: See load for eviction_policy and cache_modifier

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Supports scalar and 1~5 dimensional tensors |
| Ascend | Supports scalar and 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

#### 2.2.3 Community Constraints

1. If `pointer` is a single pointer:
   - In this case, `value` and `mask` must be scalars
   - `value` is implicitly cast to the data type of `pointer.dtype.element_ty`
   - In this case, `boundary_check` cannot be passed
2. If `pointer` is an N-Dimensional tensor:
   - `mask` and `value` are implicitly broadcast to the same shape as `pointer`
   - In this case, `boundary_check` cannot be passed
3. If `pointer` comes from `tl.make_block_ptr`:
   - In this case, `mask` must be None
   - In this case, boundary checking can be set via `boundary_check`

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation); eviction_policy and cache_modifier are not yet fully functional on NPU.

| Difference                             | Description                                                                                 | Solution                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------- |
| Generalization issues with discrete masks | Currently, discrete masks in store are handled by decomposing the store into atomic {load, select, store}, which has certain generalization issues in corner cases | Extensive generalization testing exposes issues; resolved iteratively |
| Generalization issues when used with branch and loop statements | If the computation of `pointer` and `mask` of tl.load involves complex loop and branch statements, compilation issues may occur | Extensive generalization testing exposes issues; resolved iteratively |

### 2.4 Usage

The following example implements the functionality of `torch_ldst_indirect_08_func` through the combined invocation of `triton_ldst_indirect_08_kernel` and `triton_ldst_indirect_08_func`:

```python
@triton.jit
def triton_ldst_indirect_08_kernel(
    out_ptr0, in_ptr1, in_ptr2, in_ptr3, stride_in_r,
    XS: tl.constexpr, RS: tl.constexpr
):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.arange(0, XS)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    tmp2 = tl.math.exp(tmp2)
    tmp3 = tl.load(in_ptr3 + in_idx1)
    tmp3 = tmp3 + 1 - 8
    out0_idx = in_idx0[:, None] * RS + tmp3[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)

def triton_ldst_indirect_08_func(xc, x2, xs, rs): # [8-24] ori 8 16
    nr = x2.size()[0]
    nc = xc.numel()
    stride_in_r = x2.stride()[0]
    assert nr == xs, "test only single core"
    y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
    xc1 = xc - 1
    triton_ldst_indirect_08_kernel[nr // xs, 1, 1](
        y0, xc, x2, xc1, stride_in_r, XS = xs, RS = rs)
    return y0

def torch_ldst_indirect_08_func(xr, xc, x2):
    flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
    extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
    print(extracted)
    return torch.exp(extracted)

DEV = "npu"
DTYPE = torch.float32
offset = 8
N0, N1 = 16, 32
blocksize = 8
lowdimsize = N0
assert N1 >= N0+offset, "N1 must be >= N0+offset"
assert N0 == lowdimsize, "N0 must be == lowdimsize"
xc = offset + torch.arange(0, N0, device=DEV)
xr = torch.arange(0, blocksize, device=DEV)
x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
torch_ref = torch_ldst_indirect_08_func(xr, xc, x2)
triton_cal = triton_ldst_indirect_08_func(xc, x2, blocksize, lowdimsize)
torch.testing.assert_close(triton_cal, torch_ref)
```
