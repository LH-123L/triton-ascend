# mul

## 1. OP Overview

Description: The '*' arithmetic operation; there is no tl.mul call method

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor or Number`     |     First input     |
| `y`       | `tensor or Number`     |     Second input     |

Return Value:
`tl.tensor`: The multiplication result

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range          |
| ------ | --------------- |
| GPU    | No restrictions |
| Ascend A2/A3 | No restrictions  |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape.

### 2.3 Usage Example

The following example performs the multiplication operation on the input tensors `in_ptr0, in_ptr1`:

```python
@triton.jit
def triton_mul(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

## 3.  Special Notes

Ascend A3 lacks fp64 support compared with GPU.
