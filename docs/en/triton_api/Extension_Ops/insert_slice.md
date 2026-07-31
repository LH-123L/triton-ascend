# triton.language.insert_slice

## 1. OP Overview

Description: Inserts a tensor (sub-tensor) into the specified position of another tensor, i.e., inserts a tensor into another tensor according to the offsets, sizes, and strides parameters specified by the operation.
Prototype:

```python
triton.language.insert_slice(
    ful,
    sub,
    offsets,
    sizes,
    strides,
    _builder=None,
    _generator=None
) -> tensor
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `ful`        | `tensor`          | The target tensor that receives the insertion                                                     |
| `sub`       | `tensor`    | The sub-tensor to insert; its shape must match the shape specified by the `sizes` parameter                                                        |
| `offsets`     | `tuple of ints`    | Specifies the starting offsets for insertion into the `ful` tensor (for each dimension) |
| `sizes` | `tuple of ints` | Specifies the size of the insertion region (for each dimension)                                             |
| `strides` | `tuple of ints` | Specifies the stride of the insertion region (for each dimension)                                            |
| `_builder` |- | Reserved parameter, external invocation not supported for now                                            |
| `_generator`   | -               | Reserved parameter, external invocation not supported for now                                                |

Return value:
`tensor`: The new tensor after inserting the sub-tensor

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |

#### 2.2.2 Shape Support

Tensors of any shape are supported, provided that:

1. `ful` and `sub` must have the same number of dimensions
2. The lengths of `offsets`, `sizes`, and `strides` must be the same as the number of dimensions of the tensor
3. The insertion region must not exceed the bounds of the `ful` tensor

### 2.3 Special Limitations

No special limitations

### 2.4 Usage

The following example inserts the slice computation result back into the original tensor:

```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Extract the slice
    x_sub = tl.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = tl.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    # Load the original output tensor
    output = tl.load(output_ptr + offsets, mask=mask)
    # Insert the computation result back into the original tensor
    output = tl.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. Semantic GAP

No semantic differences
