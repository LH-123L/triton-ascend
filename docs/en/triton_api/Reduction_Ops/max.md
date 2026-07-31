# triton.language.max

## 1. OP Overview

Description: Returns the maximum value along the specified dimension

```python
triton.language.max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | The input tensor data |
| `axis`   | `int` | Specifies the dimension on which to reduce; when axis = None, reduces over all axes |
| `keep_dims` | `bool` | Keeps the reduced dimension after reduction |
| `return_indices` | `bool` | If True, returns the index corresponding to the maximum value; otherwise, returns only the maximum value |
| `return_indices_tie_break_left` | `bool` | If multiple elements have the same maximum value, returns the index of the leftmost maximum value (only effective when return_indices is True) |

Return value:
`tl.tensor`: When keep_dims=True, the output has the same number of dimensions as input (the reduced axis dimension is kept as 1); when keep_dims=False, the reduced axis is removed (the number of dimensions decreases)
When return_indices = True, the returned index is of type fp32.

Supported parameter combinations:

| axis | keep_dims | return_indices | return_indices_tie_break_left  | Specification |
| ------ | ------------ | ----------------- | ----------------------------------- | ---|
|    1 |    TRUE    |      TRUE      |               TRUE                |  Supported
|    1 |    TRUE    |      TRUE      |               FALSE                | Supported
|    1 |    TRUE    |      FALSE      |               TRUE                | Supported
|    1 |    TRUE    |      FALSE      |               FALSE                | Supported
|    1 |   FALSE   |      TRUE      |               TRUE                | Supported
|    1 |   FALSE   |      TRUE      |               FALSE                | Supported
|    1 |   FALSE   |      FALSE      |               TRUE                | Supported
|    1 |   FALSE   |      FALSE      |               FALSE              | Supported
| None |    TRUE    |      TRUE      |               TRUE               | Not supported
| None |    TRUE    |      TRUE      |               FALSE              | Not supported
|  None |    TRUE    |      FALSE      |               TRUE              | Supported
| None |    TRUE    |      FALSE      |               FALSE              | Supported
| None |   FALSE   |      TRUE      |               TRUE                | Not supported
| None |   FALSE   |      TRUE      |               FALSE               | Not supported
| None |   FALSE   |      FALSE      |               TRUE              | Supported
| None |   FALSE   |      FALSE      |               FALSE             | Supported

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range         |
| -------- | ---------------------- |
| GPU    | No restrictions |
| Ascend A2/A3| No restrictions (up to 8 dimensions by default) |

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Usage

For more examples, refer to the triton-ascend code repository, ascend/examples/generalization_cases/test_max.py

```python
@triton.jit
def triton_max_1d(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.max(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```

### 2.4 Special Limitations

Ascend A3 does not support uint16, uint32, and uint64 compared to GPU
