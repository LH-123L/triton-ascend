# triton.language.range

## 1. Function Overview

`range` is an iterator that counts upward, similar to Python's `range()` function, but it allows more parameters to be passed.

```python
triton.language.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `arg1` | `int` /`constexpr`| Required | The starting value (when a single argument is given, it serves as the ending value, starting from 0) |
| `arg2` | `int`/`constexpr` | - | The ending value (not included in the range) |
| `step` | `int` /`constexpr`| `1` | An integer, the step increment for each iteration|
| `num_stages` | `int` | - | The number of pipeline stages (number of iterations executed simultaneously) |
| `loop_unroll_factor` | `int` | - | The loop unroll factor (<2 means no unrolling) |
| `disallow_acc_multi_buffer` | `bool` | `False` | Disables multi-buffer optimization of the dot operation accumulator |
| `flatten` | `bool` | `False` | Automatically flattens nested loops into a single loop |
| `warp_specialize` | `bool` | `False` | Enables warp specialization (Blackwell GPUs only) |
| `disable_licm` | `bool` | `False` | Disables loop-invariant code motion optimization |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2 Type Support

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × | × | × | × |

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).
The features related to disallow_acc_multi_buffer, flatten, warp_specialize, and disable_licm are not yet complete.

### 2.4 Usage

```python
import triton.language as tl

@triton.jit
def basic_examples():
    # Single argument: 0 to 9
    for i in tl.range(10):
        # i = 0, 1, 2, ..., 9
        pass

    # Two arguments: 2 to 9
    for i in tl.range(2, 10):
        # i = 2, 3, ..., 9
        pass

    # Three arguments: 0 to 10, step 2
    for i in tl.range(0, 10, 2):
        # i = 0, 2, 4, 6, 8
        pass
```

```python
@triton.jit
def advanced_examples():
    # Using loop optimization parameters
    for i in tl.range(0, 100, num_stages=3, loop_unroll_factor=4):
        # The number of pipeline stages is 3, and the loop unroll factor is 4
        pass

    # Nested loop flattening
    for i in tl.range(0, 10, flatten=True):
        for j in tl.range(0, 20, flatten=True):
            # The two loops will be automatically flattened into a single loop
            pass
```
