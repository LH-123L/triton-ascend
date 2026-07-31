# triton.language.device_assert

**Using `device_assert` requires setting the environment variable `TRITON_DEBUG` to a non-`0` value for it to take effect.**

## 1. Function Overview

`device_assert` is used to perform assertion checks from the device side at GPU runtime, outputting an error message if the condition is not satisfied.

```python
triton.language.device_assert(cond, msg='', _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `cond` | `bool` | Required | The condition expression to assert at runtime |
| `msg` | `str` | `''` | The error message displayed when the assertion fails |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

### 2.3 Usage

```python
import triton.language as tl

@triton.jit
def basic_device_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Basic assertion: check the program ID
    pid = tl.program_id(0)
    tl.device_assert(pid >= 0, "Program ID must be non-negative")

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)

    # Check data validity (e.g., verify that there are no negative values in the tensor)
    tl.device_assert(tl.min(x) >= 0, "All values must be non-negative")
```
