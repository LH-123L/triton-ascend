# triton.language.compile_hint

## 1. Function Overview

`compile_hint` is a compiler hint mechanism that allows users to attach metadata information to tensors. This metadata is passed to the compiler backend to guide optimization and code generation.

```python
triton.language.compile_hint(ptr, hint_name, hint_val=None, _builder=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|------|------|
| `ptr` | `tensor` | Required | The tensor object to which the hint is attached |
| `hint_name` | `str` `constexpr` | Required | The name identifier of the hint (must be a string) |
| `hint_val` | `None` `bool` `int` `constexpr` `list` | `None` | The value of the hint, supporting multiple types |
| `_builder` | - | `None` | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.3 Special Limitations

1. **hint_name must be of string type**: other types cannot be passed as the hint name
2. **The list parameter only supports integer arrays**: elements must be integers (`int` or `constexpr` integers); lists of floats or mixed types are not supported
3. **Non-intrusive design**: `compile_hint` does not change the computation semantics; it only adds metadata
4. **A tensor can be annotated multiple times**: the same tensor can have multiple hints with different names attached

### 2.4 Usage

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.compile_hint(tmp0, "hint_a")
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```
