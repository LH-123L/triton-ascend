# triton.language.atomic_add

## 1. OP Overview

Description: Atomic addition operation, which performs an atomic addition at the specified memory location
Prototype:

```python
triton.language.atomic_add(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```

It can be called as a member function of a tensor, e.g., `x.atomic_add(...)`, which is equivalent to `atomic_add(x, ...)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerDType`          | The memory location to operate on; the result of computing *pointer + val is written back to this memory                                                     |
| `val`       | `pointer.dtype.element_ty`    | The value used for the atomic addition operation (right operand)                                                        |
| `mask`     | `int1` or `tensor<int1>`, optional    | Specifies the data range to prevent out-of-bounds access |
| `sem` | `str`, optional | Specifies the memory semantics of the operation.<br>The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, can be read)<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations)                                             |
| `scope` | `str`, optional | The scope of threads that observe the synchronization effect of the atomic operation.<br>Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block) or "sys" (representing "SYSTEM").<br>We only support "gpu"                                            |
| `_semantic`   | -                 | Reserved parameter; not yet supported for external calls                                               |

Return Value:
`pointer`: tensor, the old value before the operation is performed

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | √    | √    | ×    |
| Ascend A2/A3| √    | √     | √     | √     | √      | √      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Compared with GPU, Ascend lacks support for int64 and fp64.

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

| Difference                   | Description                                                                           |
| --------------------- | ---------------------------------------------------------------------------- |
|Data type| Ascend lacks support for int64 and fp64 compared with GPU (hardware limitation) |
|sem| The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel" |
|scope               | Acceptable values are "gpu", "cta" or "sys".<br>We only support "gpu" |

### 2.4 Usage Example

The following example implements atomic addition:

```python
@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)

dtype, shape, ncore = ['int16', (32, 32), 2]
block_size = shape[0] * shape[1] / ncore
split_size = shape[0] // ncore
x0_value = 3
x0 = torch.full(shape, x0_value, dtype = getattr(torch, dtype)).npu()
x1 = torch.full((split_size, shape[1]), 2, dtype = getattr(torch, dtype)).npu()
y = torch.full((split_size, shape[1]), -10, dtype = getattr(torch, dtype)).npu()
n_elements = shape[0] * shape[1]
atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
```
