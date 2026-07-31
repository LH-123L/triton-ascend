# triton.language.add

## 1. OP Overview

Description: Addition, equivalent to the '+' arithmetic operation
Prototype:

```python
triton.language.add(x, y, sanitize_overflow: constexpr = True, _builder=None)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor or Number`     |     First input     |
| `y`       | `tensor or Number`     |     Second input     |
| `sanitize_overflow`     | `bool`    | Whether to check for overflow in integer addition; the default value is True and does not need to be specified explicitly |
| `_builder`   | -                 | Reserved parameter; not yet supported for external calls                                                |

Return Value:
`tl.tensor`: The addition result

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | --------------- |
| GPU    | No restrictions |
| Ascend | No restrictions  |

Conclusion: There is no difference between the GPU and Ascend platforms in terms of Shape.

### 2.3 Usage Example

The following example performs the addition operation on the input tensors `x_ptr, y_ptr`:

```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y # equivalent to output = tl.add(x,y)
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. Special Notes

> Ascend A3 does not support fp64 compared with GPU
