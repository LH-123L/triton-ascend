# triton.language.gather

## 1. OP Overview

Description: Performs a gather operation on the `src` tensor along the `axis` dimension according to `index`. The meaning of the gather operation is shown in the figure below:
![image](./gather.png)
Prototype:

```python
triton.language.gather(
 src: tensor,
 index: tensor,
 axis: int,
 _semantic=None
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter     | Type                | Description                                                             |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `src`         | `tensor`            | The tensor on which the gather operation is performed                   |
| `index`       | `tensor`            | The indices to gather                                                   |
| `axis`        | `int`               | The dimension along which the gather operation is performed             |
| `_semantic`   | -                   | Reserved parameter, not supported for external calls                    |

Return value: `tensor`: The result after gather

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×   | ×     | ×      | ×     | ×     | ×     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Ascend lacks support for fp64 compared to GPU (hardware limitation).

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only supports 1~5 dimensional tensors |
| Ascend A2/A3 | Only supports 1~5 dimensional tensors |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms; both support 1 to 5 dimensional tensors.

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

- Ascend lacks support for fp64 compared to GPU (hardware limitation).

### 2.4 Usage

Refer to the following example:

```python
import math
import numpy as np
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import test_common
import pytest
from test_common import TestUtils, check_ub_mem_overflow, get_dtype_size


@pytest.mark.parametrize("src_shape, indices_shape, axis", [
    ([2, 2], [4, 2], 0),
    ([3, 3], [1, 3], 0),
    ([3, 4], [4, 4], 0),
    ([4, 4], [8, 4], 0),
    ([4, 32], [4, 16], 1),
    ([4, 64], [4, 32], 1),
    ([128, 64], [128, 128], 1),
])
def test_gather(src_shape, indices_shape, axis):
    @triton.jit
    def gather_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                      src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr,
                      idx_dim1: tl.constexpr, idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                      out_dim0: tl.constexpr, out_dim1: tl.constexpr, out_stride0: tl.constexpr,
                      out_stride1: tl.constexpr):
        src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
        src = tl.load(src_ptr + src_offs)

        idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
        idx = tl.load(idx_ptr + idx_offs)

        out = tl.gather(src, idx, axis)

        out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
        tl.store(out_ptr + out_offs, out)

    def triton_gather(src: torch.Tensor, axis: int, indices: torch.Tensor):
        output = torch.empty(indices.shape, dtype=src.dtype, device=src.device)
        gather_kernel[(1, )](src, indices, output, axis,
                             src.shape[0], src.shape[1],
                             src.stride(0), src.stride(1),
                             indices.shape[0], indices.shape[1],
                             indices.stride(0), indices.stride(1),
                             output.shape[0], output.shape[1],
                             output.stride(0), output.stride(1))
        return output

    DEV = "npu"
    src = torch.randn(src_shape, device=DEV)
    indices = torch.randint(0, src.shape[axis], indices_shape, device=DEV)

    dtype_size = get_dtype_size('int32')
    if dtype_size * math.prod(src.shape) >= (TestUtils.ub_size / 8):
        print(f"dtype:int32 shape:{src.shape} mem overflow")
        return

    ref = torch.gather(src, axis, indices)
    result = triton_gather(src, axis, indices)
    torch.testing.assert_close(result, ref, rtol=0, atol=0)
```
