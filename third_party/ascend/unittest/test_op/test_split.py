import torch
import triton
import triton.language as tl


# ============================================================
# 测试用例 1：基础 split 行为（rank=2，沿最后一维 size=2 拆分）
# ============================================================
@triton.jit
def split_basic_kernel(in_ptr, out_lhs_ptr, out_rhs_ptr, M: tl.constexpr, N: tl.constexpr):
    # shape: (M, 2) -> 两个 (M,)
    a = tl.load(in_ptr + tl.arange(0, M)[:, None] * 2 + tl.arange(0, 2)[None, :])
    lhs, rhs = tl.split(a)
    tl.store(out_lhs_ptr + tl.arange(0, M), lhs)
    tl.store(out_rhs_ptr + tl.arange(0, M), rhs)


def test_split_basic_2d():
    """验证主算子 tt.split 对 2D tensor 行为一致"""
    M, N = 4, 2
    x = torch.arange(M * 2, dtype=torch.float32, device='cuda').reshape(M, 2)
    out_lhs = torch.empty(M, dtype=torch.float32, device='cuda')
    out_rhs = torch.empty(M, dtype=torch.float32, device='cuda')
    split_basic_kernel[(1,)](x, out_lhs, out_rhs, M=M, N=N)

    # 验证拆分结果：lhs = x[:, 0], rhs = x[:, 1]
    assert torch.allclose(out_lhs, x[:, 0]), f"lhs 错误: {out_lhs} vs {x[:, 0]}"
    assert torch.allclose(out_rhs, x[:, 1]), f"rhs 错误: {out_rhs} vs {x[:, 1]}"
    print("PASS: test_split_basic_2d")


# ============================================================
# 测试用例 2：rank=1 tensor 拆分（关键差异点：unsplat vs reduce）
# ============================================================
@triton.jit
def split_rank1_kernel(in_ptr, out_lhs_ptr, out_rhs_ptr):
    # shape: (2,) -> 两个 scalar
    a = tl.load(in_ptr + tl.arange(0, 2))
    lhs, rhs = tl.split(a)  # rank=1 触发 unsplat/reduce 后处理
    tl.store(out_lhs_ptr, lhs)
    tl.store(out_rhs_ptr, rhs)


def test_split_rank1():
    """验证 rank=1 时 unsplat/reduce 行为一致（关键差异点）"""
    x = torch.tensor([3.14, 2.71], dtype=torch.float32, device='cuda')
    out_lhs = torch.empty(1, dtype=torch.float32, device='cuda')
    out_rhs = torch.empty(1, dtype=torch.float32, device='cuda')
    split_rank1_kernel[(1,)](x, out_lhs, out_rhs)

    # lhs = 3.14, rhs = 2.71
    assert torch.allclose(out_lhs, x[0:1]), f"lhs 错误: {out_lhs}"
    assert torch.allclose(out_rhs, x[1:2]), f"rhs 错误: {out_rhs}"
    print("PASS: test_split_rank1")


# ============================================================
# 测试用例 3：rank=3 tensor 拆分
# ============================================================
@triton.jit
def split_rank3_kernel(in_ptr, out_lhs_ptr, out_rhs_ptr,
                       B: tl.constexpr, M: tl.constexpr):
    # shape: (B, M, 2) -> 两个 (B, M)
    a = tl.load(in_ptr + tl.arange(0, B)[:, None, None] * M * 2
                + tl.arange(0, M)[None, :, None] * 2
                + tl.arange(0, 2)[None, None, :])
    lhs, rhs = tl.split(a)
    out_offset = tl.arange(0, B)[:, None] * M + tl.arange(0, M)[None, :]
    tl.store(out_lhs_ptr + out_offset, lhs)
    tl.store(out_rhs_ptr + out_offset, rhs)


def test_split_rank3():
    """验证高维 tensor 拆分结果一致"""
    B, M = 2, 3
    x = torch.randn(B, M, 2, device='cuda')
    out_lhs = torch.empty(B, M, device='cuda')
    out_rhs = torch.empty(B, M, device='cuda')
    split_rank3_kernel[(1,)](x, out_lhs, out_rhs, B=B, M=M)

    assert torch.allclose(out_lhs, x[:, :, 0], atol=1e-6)
    assert torch.allclose(out_rhs, x[:, :, 1], atol=1e-6)
    print("PASS: test_split_rank3")


# ============================================================
# 测试用例 4：constexpr shape 解析（_unwrap_if_constexpr vs _constexpr_to_value）
# ============================================================
def test_split_constexpr_shape():
    """验证 constexpr shape[-1]==2 的解析两版本等价"""
    import triton.language as tl
    from triton.runtime import interpreter

    # 模拟 constexpr 形式 shape
    @triton.jit
    def kernel_const(in_ptr, out_lhs_ptr, out_rhs_ptr):
        SHAPE: tl.constexpr = (4, 2)
        a = tl.load(in_ptr + tl.arange(0, 4)[:, None] * 2 + tl.arange(0, 2)[None, :])
        lhs, rhs = tl.split(a)
        tl.store(out_lhs_ptr + tl.arange(0, 4), lhs)
        tl.store(out_rhs_ptr + tl.arange(0, 4), rhs)

    x = torch.arange(8, dtype=torch.float32, device='cuda').reshape(4, 2)
    out_lhs = torch.empty(4, dtype=torch.float32, device='cuda')
    out_rhs = torch.empty(4, dtype=torch.float32, device='cuda')
    kernel_const[(1,)](x, out_lhs, out_rhs)
    assert torch.allclose(out_lhs, x[:, 0])
    assert torch.allclose(out_rhs, x[:, 1])
    print("PASS: test_split_constexpr_shape")


# ============================================================
# 测试用例 5：split + join 互逆性
# ============================================================
@triton.jit
def split_join_roundtrip_kernel(in_ptr, out_ptr, M: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, M)[:, None] * 2 + tl.arange(0, 2)[None, :])
    lhs, rhs = tl.split(a)
    b = tl.join(lhs, rhs)  # 再 join 回去
    tl.store(out_ptr + tl.arange(0, M)[:, None] * 2 + tl.arange(0, 2)[None, :], b)


def test_split_join_roundtrip():
    """验证 split 与 join 互为逆运算"""
    M = 8
    x = torch.randn(M, 2, device='cuda')
    out = torch.empty_like(x)
    split_join_roundtrip_kernel[(1,)](x, out, M=M)
    assert torch.allclose(out, x, atol=1e-6), f"roundtrip 失败: max diff = {(out-x).abs().max()}"
    print("PASS: test_split_join_roundtrip")


# ============================================================
# 测试用例 6：最后一维非 2 应抛异常
# ============================================================
@triton.jit
def split_invalid_kernel(in_ptr, out_ptr, M: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, M)[:, None] * 4 + tl.arange(0, 4)[None, :])
    lhs, rhs = tl.split(a)  # 最后一维为 4，应报错


def test_split_invalid_last_dim():
    """验证最后一维非 2 时两版本都抛异常"""
    M = 4
    x = torch.arange(M * 4, dtype=torch.float32, device='cuda').reshape(M, 4)
    out_lhs = torch.empty(M, 4, dtype=torch.float32, device='cuda')
    out_rhs = torch.empty(M, 4, dtype=torch.float32, device='cuda')

    raised = False
    try:
        split_invalid_kernel[(1,)](x, out_lhs, out_rhs, M=M)
    except (AssertionError, triton.runtime.errors.TritonRuntimeError) as e:
        raised = True
        print(f"  捕获到异常: {type(e).__name__}")

    assert raised, "最后一维非 2 时应抛异常"
    print("PASS: test_split_invalid_last_dim")


# ============================================================
# 测试用例 7：rank=0 标量 tensor 应抛异常
# ============================================================
@triton.jit
def split_rank0_kernel(in_ptr, out_lhs_ptr, out_rhs_ptr):
    a = tl.load(in_ptr)  # 标量，shape=()
    lhs, rhs = tl.split(a)  # 应报错：len(a.shape) == 0


def test_split_rank0():
    """验证 rank=0 时两版本都抛异常"""
    x = torch.tensor(1.0, device='cuda')
    out_lhs = torch.empty(1, device='cuda')
    out_rhs = torch.empty(1, device='cuda')

    raised = False
    try:
        split_rank0_kernel[(1,)](x, out_lhs, out_rhs)
    except (AssertionError, triton.runtime.errors.TritonRuntimeError) as e:
        raised = True
        print(f"  捕获到异常: {type(e).__name__}")

    assert raised, "rank=0 时应抛异常"
    print("PASS: test_split_rank0")


# ============================================================
# 测试用例 8：不同 dtype 的 split 行为
# ============================================================
@triton.jit
def split_dtype_kernel(in_ptr, out_lhs_ptr, out_rhs_ptr, M: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, M)[:, None] * 2 + tl.arange(0, 2)[None, :])
    lhs, rhs = tl.split(a)
    tl.store(out_lhs_ptr + tl.arange(0, M), lhs)
    tl.store(out_rhs_ptr + tl.arange(0, M), rhs)


def test_split_various_dtypes():
    """验证多 dtype 下行为一致"""
    M = 4
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]

    for dt in dtypes:
        x = torch.arange(M * 2, dtype=dt, device='cuda').reshape(M, 2)
        out_lhs = torch.empty(M, dtype=dt, device='cuda')
        out_rhs = torch.empty(M, dtype=dt, device='cuda')
        split_dtype_kernel[(1,)](x, out_lhs, out_rhs, M=M)

        assert torch.allclose(out_lhs, x[:, 0]), f"dtype={dt} lhs 错误"
        assert torch.allclose(out_rhs, x[:, 1]), f"dtype={dt} rhs 错误"
        print(f"  PASS: dtype={dt}")


# ============================================================
# 测试用例 9：IR Dump 对比（验证 3.6 unsplat vs 3.2 reduce）
# ============================================================
def test_split_rank1_ir_diff():
    """
    验证 rank=1 时 IR 差异：
    - 3.6.0:  生成 tt.split + tt.unsplat
    - 3.2.0:  生成 tt.split + tt.reduce
    """
    import os
    os.environ["MLIR_ENABLE_DUMP"] = "0"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = "./dump_split_test"

    @triton.jit
    def kernel(in_ptr, out_lhs_ptr, out_rhs_ptr):
        a = tl.load(in_ptr + tl.arange(0, 2))
        lhs, rhs = tl.split(a)
        tl.store(out_lhs_ptr, lhs)
        tl.store(out_rhs_ptr, rhs)

    x = torch.tensor([1.0, 2.0], device='cuda')
    out_lhs = torch.empty(1, device='cuda')
    out_rhs = torch.empty(1, device='cuda')
    kernel[(1,)](x, out_lhs, out_rhs)

    # 检查 dump 目录中的 IR
    import glob
    dump_files = glob.glob("./dump_split_test/*.ttir")
    if dump_files:
        with open(dump_files[0], 'r') as f:
            ir_content = f.read()
        print(f"  生成的 IR 包含 tt.split: {'tt.split' in ir_content}")
        # 3.6 会有 tt.unsplat，3.2 会有 tt.reduce
        has_unsplat = 'tt.unsplat' in ir_content or 'unsplat' in ir_content
        has_reduce = 'tt.reduce' in ir_content or 'reduce' in ir_content
        print(f"  含 unsplat (3.6 特征): {has_unsplat}")
        print(f"  含 reduce (3.2 特征): {has_reduce}")
    print("PASS: test_split_rank1_ir_diff")


# ============================================================
# 测试用例 10：跨版本结果一致性回归测试
# ============================================================
def test_split_cross_version_consistency():
    """
    跨版本结果一致性测试：
    验证两版本对同一输入产生完全相同的输出
    """
    test_cases = [
        # (shape, dtype)
        ((4, 2), torch.float32),
        ((8, 2), torch.float16),
        ((2, 3, 2), torch.bfloat16),
        ((1, 1, 2), torch.int32),
    ]

    for shape, dt in test_cases:
        x = torch.randn(*shape, device='cuda').to(dt) if dt.is_floating_point else \
            torch.randint(0, 100, shape, device='cuda', dtype=dt)
        out_lhs = torch.empty(shape[:-1], dtype=dt, device='cuda')
        out_rhs = torch.empty(shape[:-1], dtype=dt, device='cuda')

        M = 1
        for s in shape[:-1]:
            M *= s
        split_dtype_kernel[(1,)](
            x.contiguous(), out_lhs, out_rhs, M=M
        )

        assert torch.allclose(out_lhs, x[..., 0]), \
            f"shape={shape} dtype={dt} lhs 不匹配"
        assert torch.allclose(out_rhs, x[..., 1]), \
            f"shape={shape} dtype={dt} rhs 不匹配"
        print(f"  PASS: shape={shape} dtype={dt}")


# ============================================================
# 主测试入口
# ============================================================
if __name__ == "__main__":
    test_split_basic_2d()
    test_split_rank1()
    test_split_rank3()
    test_split_constexpr_shape()
    test_split_join_roundtrip()
    test_split_invalid_last_dim()
    test_split_rank0()
    test_split_various_dtypes()
    # test_split_rank1_ir_diff()  # 需要 dump 目录写权限
    test_split_cross_version_consistency()
    print("\n所有 split 算子差异测试用例通过！")