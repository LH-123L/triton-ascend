# triton.language.inline_asm_elementwise

## 1. Function Overview

`inline_asm_elementwise` is used to execute inline assembly code in Triton kernels to perform element-wise operations on tensors.

```python
triton.language.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `asm` | `str` | Required | The assembly code to execute; it must match the assembly format of the target platform |
| `constraints` | `str` | Required | Assembly constraints in LLVM format |
| `args` | `tensor` | Required | Input tensors, whose values are passed to the assembly block |
| `dtype` | `dtype` / `Sequence[dtype]` | Required | The element type of the returned tensor (can be a single type or a tuple of types) |
| `is_pure` | `bool` | Required | If True, the compiler assumes the assembly block has no side effects |
| `pack` | `int` | Required | The number of elements processed per inline assembly call |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ |×|   ✓  | × | × | ×  |

Compared with GPU, Ascend lacks support for the input tensor types uint8, uint16, uint32, uint64, fp16, fp64, bf16, and bool.

### 2.3 Usage

```python
import triton.language as tl
@triton.jit
def triton_asm_add(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="""
        ADD.s64 $0, $1, $2
        """,
        constraints=(
            "=l,l,l"
        ),
        args=[x, y],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. Semantic GAP

1. The registers of the inline assembly only support `int64(s64)` and `float32(f32)`.
2. The constraint restriction only supports `l`.
3. Currently, only one-dimensional input tensors are supported; computing high-dimensional tensors requires unrolling.
