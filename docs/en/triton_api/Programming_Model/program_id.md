# triton.language.program_id

## 1. OP Overview

Description: Returns the ID of the current program instance along the given axis.
Function prototype:

```python
triton.language.program_id(axis)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `axis` | `int` | The axis of the 3D launch grid. Must be 0, 1, or 2 |

Return value:
A tl.tensor of the axis value

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | √ | × | × | × | × | × | × | × | × | × | × |
| Ascend A2/A3 | × | × | √ | × | × | × | × | × | × | × | × | × | × |

#### 2.2.2 Shape Support

Not applicable

### 2.3 Special Limitations

None

### 2.4 Usage

It is used in triton kernels to obtain the PID

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ...

    tl.store(output_ptr + idx, ret)
```
