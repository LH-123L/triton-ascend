# triton.language.static_assert

## 1. Function Overview

`static_assert` is used to assert whether a condition holds at compile time; compilation fails if the condition is not satisfied. This is a compile-time checking tool and does not require setting debug environment variables.

```python
triton.language.static_assert(cond, msg='', _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `cond` | `bool` | Required | The condition expression to assert at compile time |
| `msg` | `str` | `''` | The error message displayed when the assertion fails |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

**Note:** The type of the value in the `cond` statement must be `constexpr`.

### 2.3 Usage

```python
import triton.language as tl

@triton.jit
def basic_static_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Basic assertion: check whether BLOCK_SIZE is a power of 2
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    # Assertion with a custom error message
    tl.static_assert(BLOCK_SIZE >= 64, "BLOCK_SIZE must be at least 64 for performance")

    # A non-constant in the static_assert condition causes a compilation error
    # val = tl.load(x_ptr)
    # tl.static_assert(val <= 64)
```
