# triton.language.get_element

## 1. OP Overview

Description: Reads a single element from the input tensor based on the given indices.
Prototype:

```python
triton.language.get_element(
    src,
    indice,
    _builder=None,
    _generator=None
) -> scalar
```

It can be called as a member function of a tensor, e.g., `x.get_element(...)`, which is equivalent to `get_element(x, ...)`.

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter           | Type                | Description                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `src`        | `tensor`          | The source tensor to be accessed                                                     |
| `indice`       | `tuple of ints` or `tuple of tensors`    | The indices used to specify the position of the element                                                    |
| `_builder` |- | Reserved parameter, external invocation not supported for now                                            |
| `_generator`   | -               | Reserved parameter, external invocation not supported for now                                                |

Return value:
`scalar`: A scalar value of the same type as the elements of the `src` tensor

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |

#### 2.2.2 Shape Support

Tensors of any shape are supported, provided that:
the length of `indice` must be the same as the number of dimensions of the `src` tensor.

### 2.3 Special Limitations

No special limitations

### 2.4 Usage

The following example demonstrates the use of get_element:

```python
@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim,
                                g_stride: tl.constexpr, indice_length: tl.constexpr,
                                g_block: tl.constexpr, g_block_sub: tl.constexpr,
                                other_block: tl.constexpr):
    """
    Manual implementation using tl.get_element and tl.insert_slice.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Manual gather: iterate over each index
            for i in range(0, g_block_sub):
                gather_offset = tl.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = tl.insert_slice(tmp_buf, val[None, :],
                                          offsets=(i, 0), sizes=(1, other_block), strides=(1, 1))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```

## 3. Semantic GAP

No semantic differences
