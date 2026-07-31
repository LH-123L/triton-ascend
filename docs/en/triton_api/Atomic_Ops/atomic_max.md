# triton.language.atomic_max

## 1. OP Overview

Description: Atomic maximum operation, which performs an atomic maximum operation at the specified memory location
Prototype:

```python
triton.language.atomic_max(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```

It can be called as a member function of a tensor, e.g., `x.atomic_max(...)`, which is equivalent to `atomic_max(x, ...)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerDType`          | The memory location to operate on; the result of computing max(*pointer , val) is written back to this memory                                                     |
| `val`       | `pointer.dtype.element_ty`    | The value used to compare with the target memory                                                        |
| `mask`     | `int1` or `tensor<int1>`, optional    | Specifies the data range to prevent out-of-bounds access |
| `sem` | `str`, optional | Specifies the memory semantics of the operation.<br>The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, can be read)<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations)                                             |
| `scope` | `str`, optional | The scope of threads that observe the synchronization effect of the atomic operation.<br>Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block) or "sys" (representing "SYSTEM").<br>We only support "gpu"                                            |
| `_semantic`   | -                 | Reserved parameter; not yet supported for external calls                                                |

Return Value:
`pointer`: tensor, the old value before the operation is performed

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU     | ×     | ×      |  √     | ×     | ×      | ×      | ×      |√    | ×     | √    | ×      | ×      | ×     |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Compared with GPU, Ascend lacks support for int64.

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

| Difference                   | Description                                                                           |
| --------------------- | ---------------------------------------------------------------------------- |
|Data type| Ascend lacks support for int64 compared with GPU (hardware limitation) |
|sem| The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel" |
|scope               | Acceptable values are "gpu", "cta" or "sys".<br>We only support "gpu" |

### 2.4 Usage Example

The following example implements atomic maximum:

```python
@triton.jit
def triton_atomic_max(
    in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_max(out_ptr0 + (x1), tmp0, xmask)
```
