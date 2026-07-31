# triton.language.atomic_and

## 1. OP Overview

Description: Atomic logical AND operation, which performs a logical AND at the specified memory location
Prototype:

```python
triton.language.atomic_and(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```

It can be called as a member function of a tensor, e.g., `x.atomic_and(...)`, which is equivalent to `atomic_and(x, ...)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerDType`          | The memory location to operate on; the result of computing *pointer & val is written back to this memory                                                     |
| `val`       | `pointer.dtype.element_ty`    | The value used for the atomic AND operation (right operand)                                                        |
| `mask`     | `int1` or `tensor<int1>`, optional    | Specifies the data range to prevent out-of-bounds access |
| `sem` | `str`, optional | Specifies the memory semantics of the operation.<br>The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, can be read)<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations)                                             |
| `scope` | `str`, optional | The scope of threads that observe the synchronization effect of the atomic operation.<br>Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block) or "sys" (representing "SYSTEM").<br>We only support "gpu"                                            |
| `_semantic`   | -                 | Reserved parameter; not yet supported for external calls                                              |

Return Value:
`pointer`: tensor, the old value before the operation is performed

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU   | ×     | ×      |√     | ×     | ×      | ×      | ×      | √     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3| √    | √     | √     | √     | √      | √      | √      | √     | ×    | ×    | ×    | ×    | ×    |

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Restrictions

> Features missing relative to the community that cannot be implemented

| Difference                   | Description                                                                           | Solution                          |
| --------------------- | ---------------------------------------------------------------------------- | ----------------------------- |
|sem| The values accepted by the community official configuration are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE") and "relaxed".<br>We only support "acq_rel" | To be developed |
|scope               | Acceptable values are "gpu", "cta" or "sys".<br>We only support "gpu" | To be developed |

### 2.4 Usage Example

The following example implements atomic AND:

```python
@triton.jit
def atomic_and(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_and(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)

dtype, shape, ncore = ['int32', (32, 32), 2]
block_size = shape[0] * shape[1] // ncore
split_size = shape[0] // ncore
val = torch.randint(low=0, high=10, size=shape, dtype=getattr(torch, dtype)).npu()
pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=getattr(torch, dtype)).npu()
pointer_old = torch.full_like(pointer, -10).npu()
n_elements = shape[0] * shape[1]
atomic_and[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
```
