# triton.language.device_print

## 1. Function Overview

`device_print` is used to print information from the device side at NPU runtime. Unlike `static_print`, this outputs information in real time during kernel execution. The first parameter must be a `string`, and the following parameters must be `scalars` or `tensors`. **Using `device_print` requires setting the environment variable `TRITON_DEVICE_PRINT` to `True`.**

```python
triton.language.device_print(prefix, *args, hex=False, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|--------|----------|
| `prefix` | `str` | Required | The prefix string printed before the values |
| `args` | `tensor`/`scalar` | Required | The values to print, can be any tensors or scalars |
| `hex` | `bool` | `False` | Whether to print all values in hexadecimal format |
| `_semantic` | - | - | Reserved parameter, external invocation not supported for now |

### 2.2.1 Data Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | Only 1~5-dimensional tensors supported |
| Ascend | Only 1~5-dimensional tensors supported |

### 2.3 Special Limitations

> Capability missing relative to the community and cannot be implemented

Compared with GPU, Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 (hardware limitation).

### 2.4 Usage

**Note**: The `prefix` string prefix must be provided when using `device_print`; otherwise, a compilation error will occur.

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0,3)
    idy = tl.arange(0,4)
    offset = idx[:,None] * 4 + idy[None,:]
    val = tl.load(x_ptr + offset)
    # Print the value of the two-dimensional tensor val
    tl.device_print("val:",val)
```
