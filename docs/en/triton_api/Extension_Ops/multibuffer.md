# triton.language.multibuffer

## 1. OP Overview

Description: Sets up multi-buffering for a tensor, allowing the compiler to create multiple copies of the same tensor.
Prototype:

```python
triton.language.multibuffer(
    src,
    size,
    _builder=None
) -> None
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `src`        | `tensor`          | The source tensor for which multi-buffering is set up                                                     |
| `size`       | `int` or `constexpr`    | The number of buffer copies to create                                                        |
| `_builder` |- | Reserved parameter, external invocation not supported for now                                            |

Return value:
`None`: This operation is a compile-time hint that does not return a value at runtime and only affects the compiler's optimization behavior.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 |  bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | √    |

#### 2.2.2 Shape Support

Tensors of any shape are supported.

### 2.3 Special Limitations

| Restricted Parameter                   | Description                                                                           |
| --------------------- | ---------------------------------------------------------------------------- |
|`size` | The current implementation only supports `size` being `2`. |

### 2.4 Usage

The following example shows how to set up multi-buffering for the tensor `tmp0` in a kernel and use it together with other compile hints:

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        # Set up double buffering for tmp0
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```
