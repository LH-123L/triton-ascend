# triton.language.num_programs

## 1. OP Overview

Description: Returns the number of program instances launched along the given axis
Function prototype:

```python
triton.language.num_programs(axis)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `axis` | `int` | The axis of the 3D launch grid. Must be 0, 1, or 2. |

Return value:
A tl.tensor of the number of launched program instances

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

For an example, refer to [test_3Dgrid.py](../../../../third_party/ascend/unittest/pytest_ut/test_3Dgrid.py)

```python
@triton.jit
def triton_(in_ptr0, out_ptr0, x0_numel, r1_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
            block_id_threshold: tl.constexpr, XBLOCK1: tl.constexpr, num_core: tl.constexpr):
    RBLOCK: tl.constexpr = 64

    block_idx=tl.program_id(0)*tl.num_programs(1)*tl.num_programs(2)+tl.program_id(1)*tl.num_programs(2)+tl.program_id(2)
    if (block_idx < block_id_threshold):
        offset = block_idx * XBLOCK
        loops1 = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB # 32+23 / 24 = 2
        upper = offset + XBLOCK
    else:
        offset = block_id_threshold * XBLOCK + (block_idx - block_id_threshold) * XBLOCK1 #pid=34 offset = 9*32 + (34-9)*24 = 888
        loops1 = (XBLOCK1 + XBLOCK_SUB - 1) // XBLOCK_SUB #1
        if (block_idx ==num_core -1):
            upper = x0_numel
        else:
            upper = offset + XBLOCK1 # 912

    base1 = tl.arange(0, XBLOCK_SUB)
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = (r1_numel + RBLOCK - 1) // RBLOCK
    for loop1 in range(loops1):
        x = offset + (loop1 * XBLOCK_SUB) + base1
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        xmask = x0 < upper
        r1_prime = base2[:, None]
        rindex = base2
        r1 = base2[None, :]
        rmask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask,other=0.0)

        tmp1 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
        tmp2_tmp = tl.sum(tmp1,1)
        tmp2 = tmp2_tmp.reshape(XBLOCK_SUB,1)

        tl.store(out_ptr0 + (x0), tmp2, xmask)

```
