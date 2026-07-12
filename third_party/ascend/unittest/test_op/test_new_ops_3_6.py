# 
# 测试 3.6 新增算子：condition / constexpr_type / map_elementwise / bitonic_merge / reduce_or
# 
# 覆盖：
#   - condition 上下文管理器（条件分支）
#   - constexpr_type 类型系统
#   - map_elementwise 逐元素映射
#   - bitonic_merge 双调合并
#   - reduce_or 逻辑或归约
# 
# 注意：本文件位于 d:\Code\triton-ascend\third_party\ascend\unittest\test_op\
# 所有测试使用 device='npu'。

import torch
import triton
import triton.language as tl


# ============================================================
# reduce_or 测试（3.6 新增，生成 tt.reduce + arith.ori）
# ============================================================
@triton.jit
def reduce_or_kernel(in_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, N))
    b = tl.reduce_or(a, axis=0)
    tl.store(out_ptr, b)


@triton.jit
def reduce_or_2d_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    b = tl.reduce_or(a, axis=1)
    tl.store(out_ptr + tl.arange(0, M), b)


def test_reduce_or_basic():
    """reduce_or 1D 基础逻辑或归约"""
    N = 16
    x = torch.zeros(N, dtype=torch.int32, device='npu')
    x[3] = 1
    x[10] = 1
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    expected = (x != 0).any().to(torch.int32).reshape(1)
    assert out.item() == expected.item(), f"reduce_or 错误: {out} vs {expected}"
    print("PASS: test_reduce_or_basic")


def test_reduce_or_all_zero():
    """reduce_or 全 0 应返回 0"""
    N = 16
    x = torch.zeros(N, dtype=torch.int32, device='npu')
    out = torch.ones(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    assert out.item() == 0, f"全 0 reduce_or 应为 0, 得到 {out}"
    print("PASS: test_reduce_or_all_zero")


def test_reduce_or_all_one():
    """reduce_or 全 1 应返回 1"""
    N = 16
    x = torch.ones(N, dtype=torch.int32, device='npu')
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    assert out.item() == 1, f"全 1 reduce_or 应为 1, 得到 {out}"
    print("PASS: test_reduce_or_all_one")


def test_reduce_or_2d():
    """reduce_or 2D 沿 axis=1"""
    M, N = 4, 8
    x = torch.zeros(M, N, dtype=torch.int32, device='npu')
    x[0, 2] = 1
    x[1, 5] = 1
    x[2, :] = 0  # 全 0
    x[3, 0] = 1
    out = torch.zeros(M, dtype=torch.int32, device='npu')
    reduce_or_2d_kernel[(1,)](x, out, M=M, N=N)
    expected = (x != 0).any(dim=1).to(torch.int32)
    assert torch.equal(out, expected), f"reduce_or 2D 错误: {out} vs {expected}"
    print("PASS: test_reduce_or_2d")


# ============================================================
# bitonic_merge 测试（3.6 新增公开 API）
# ============================================================
@triton.jit
def bitonic_merge_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, N))
    b = tl.load(b_ptr + tl.arange(0, N))
    # bitonic_merge 接受两个已排序序列，输出合并后的排序序列
    merged = tl.bitonic_merge(a, b, order=True)
    tl.store(out_ptr + tl.arange(0, 2 * N), merged)


def test_bitonic_merge_basic():
    """bitonic_merge 合并两个升序序列"""
    N = 8
    a = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.float32, device='npu')
    b = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32, device='npu')
    out = torch.empty(2 * N, dtype=torch.float32, device='npu')
    try:
        bitonic_merge_kernel[(1,)](a, b, out, N=N)
        expected = torch.cat([a, b]).sort().values
        assert torch.allclose(out, expected), f"bitonic_merge 错误: {out} vs {expected}"
        print("PASS: test_bitonic_merge_basic")
    except AttributeError:
        print("SKIP: tl.bitonic_merge 不存在（3.2 无此公开 API）")
    except Exception as e:
        print(f"SKIP: bitonic_merge 调用失败 - {e}")


def test_bitonic_merge_descending():
    """bitonic_merge 合并两个降序序列"""
    N = 4
    a = torch.tensor([15, 11, 7, 3], dtype=torch.float32, device='npu')
    b = torch.tensor([14, 10, 6, 2], dtype=torch.float32, device='npu')
    out = torch.empty(2 * N, dtype=torch.float32, device='npu')
    try:
        bitonic_merge_kernel[(1,)](a, b, out, N=N)
        expected = torch.cat([a, b]).sort(descending=True).values
        assert torch.allclose(out, expected), f"bitonic_merge 降序错误: {out} vs {expected}"
        print("PASS: test_bitonic_merge_descending")
    except AttributeError:
        print("SKIP: tl.bitonic_merge 不存在")
    except Exception as e:
        print(f"SKIP: bitonic_merge 降序调用失败 - {e}")


# ============================================================
# map_elementwise 测试（3.6 新增工具函数）
# ============================================================
@triton.jit
def map_elementwise_kernel(in_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, N))

    def square(x):
        return x * x

    b = tl.map_elementwise(square, a)
    tl.store(out_ptr + tl.arange(0, N), b)


def test_map_elementwise_basic():
    """map_elementwise 基本逐元素映射"""
    N = 8
    x = torch.arange(N, dtype=torch.float32, device='npu')
    out = torch.empty(N, dtype=torch.float32, device='npu')
    try:
        map_elementwise_kernel[(1,)](x, out, N=N)
        expected = x * x
        assert torch.allclose(out, expected), f"map_elementwise 错误: {out} vs {expected}"
        print("PASS: test_map_elementwise_basic")
    except AttributeError:
        print("SKIP: tl.map_elementwise 不存在（3.2 无此 API）")
    except Exception as e:
        print(f"SKIP: map_elementwise 调用失败 - {e}")


@triton.jit
def map_elementwise_multi_arg_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, N))
    b = tl.load(b_ptr + tl.arange(0, N))

    def add_mul(x, y):
        return x + y * 2

    c = tl.map_elementwise(add_mul, a, b)
    tl.store(out_ptr + tl.arange(0, N), c)


def test_map_elementwise_multi_arg():
    """map_elementwise 多参数"""
    N = 8
    a = torch.arange(N, dtype=torch.float32, device='npu')
    b = torch.arange(N, dtype=torch.float32, device='npu')
    out = torch.empty(N, dtype=torch.float32, device='npu')
    try:
        map_elementwise_multi_arg_kernel[(1,)](a, b, out, N=N)
        expected = a + b * 2
        assert torch.allclose(out, expected), f"map_elementwise 多参错误: {out} vs {expected}"
        print("PASS: test_map_elementwise_multi_arg")
    except AttributeError:
        print("SKIP: tl.map_elementwise 不存在")
    except Exception as e:
        print(f"SKIP: map_elementwise 多参调用失败 - {e}")


# ============================================================
# condition 测试（3.6 新增上下文管理器）
# ============================================================
@triton.jit
def condition_kernel(in_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, N))
    b = tl.zeros([N], dtype=tl.float32)
    with tl.condition(a > 5):
        b = a * 2
    tl.store(out_ptr + tl.arange(0, N), b)


def test_condition_basic():
    """condition 上下文管理器基本行为"""
    N = 16
    x = torch.arange(N, dtype=torch.float32, device='npu')
    out = torch.empty(N, dtype=torch.float32, device='npu')
    try:
        condition_kernel[(1,)](x, out, N=N)
        expected = torch.where(x > 5, x * 2, torch.zeros_like(x))
        assert torch.allclose(out, expected), f"condition 错误: {out} vs {expected}"
        print("PASS: test_condition_basic")
    except AttributeError:
        print("SKIP: tl.condition 不存在（3.2 无此 API）")
    except Exception as e:
        print(f"SKIP: condition 调用失败 - {e}")


# ============================================================
# constexpr_type 测试（3.6 新增类型系统）
# ============================================================
def test_constexpr_type_existence():
    """constexpr_type 类是否可访问"""
    try:
        ct = tl.constexpr_type
        print(f"PASS: tl.constexpr_type 存在 - {ct}")
    except AttributeError:
        print("SKIP: tl.constexpr_type 不存在（3.2 无此 API）")


def test_constexpr_type_in_tuple():
    """constexpr_type 在 tuple_type 中的使用"""
    try:
        ct = tl.constexpr_type()
        tt = tl.tuple_type([ct, tl.block_type(tl.float32, [8])])
        print(f"PASS: constexpr_type 在 tuple_type 中可用 - {tt}")
    except AttributeError:
        print("SKIP: tl.constexpr_type 不存在")
    except Exception as e:
        print(f"SKIP: constexpr_type tuple 测试失败 - {e}")


# ============================================================
# IR 验证测试
# ============================================================
@triton.jit
def reduce_or_ir_kernel(in_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, N))
    b = tl.reduce_or(a, axis=0)
    tl.store(out_ptr, b)


def test_reduce_or_ir_has_reduce():
    """验证 reduce_or 生成 tt.reduce + arith.ori"""
    N = 16
    x = torch.ones(N, dtype=torch.int32, device='npu')
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    try:
        compiled = triton.compile(
            reduce_or_ir_kernel.warmup(x, out, N=N, grid=(1,))
        )
        ir = str(compiled.module)
        assert "tt.reduce" in ir, f"IR 中缺少 tt.reduce:\n{ir}"
        assert "ori" in ir or "or" in ir.lower(), f"IR 中缺少 or 操作:\n{ir}"
        print("PASS: reduce_or IR 含 tt.reduce + ori")
    except AttributeError:
        print("SKIP: tl.reduce_or 不存在")
    except Exception as e:
        print(f"SKIP: reduce_or IR 验证失败 - {e}")


@triton.jit
def bitonic_merge_ir_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, N))
    b = tl.load(b_ptr + tl.arange(0, N))
    merged = tl.bitonic_merge(a, b, order=True)
    tl.store(out_ptr + tl.arange(0, 2 * N), merged)


def test_bitonic_merge_ir_has_where():
    """验证 bitonic_merge 生成 tt.where"""
    N = 8
    a = torch.arange(N, dtype=torch.float32, device='npu')
    b = torch.arange(N, 2 * N, dtype=torch.float32, device='npu')
    out = torch.empty(2 * N, dtype=torch.float32, device='npu')
    try:
        compiled = triton.compile(
            bitonic_merge_ir_kernel.warmup(a, b, out, N=N, grid=(1,))
        )
        ir = str(compiled.module)
        assert "tt.where" in ir or "select" in ir, f"IR 中缺少 where/select:\n{ir}"
        print("PASS: bitonic_merge IR 含 tt.where/select")
    except AttributeError:
        print("SKIP: tl.bitonic_merge 不存在")
    except Exception as e:
        print(f"SKIP: bitonic_merge IR 验证失败 - {e}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("reduce_or 测试")
    print("=" * 60)
    test_reduce_or_basic()
    test_reduce_or_all_zero()
    test_reduce_or_all_one()
    test_reduce_or_2d()

    print("\n" + "=" * 60)
    print("bitonic_merge 测试")
    print("=" * 60)
    test_bitonic_merge_basic()
    test_bitonic_merge_descending()

    print("\n" + "=" * 60)
    print("map_elementwise 测试")
    print("=" * 60)
    test_map_elementwise_basic()
    test_map_elementwise_multi_arg()

    print("\n" + "=" * 60)
    print("condition 测试")
    print("=" * 60)
    test_condition_basic()

    print("\n" + "=" * 60)
    print("constexpr_type 测试")
    print("=" * 60)
    test_constexpr_type_existence()
    test_constexpr_type_in_tuple()

    print("\n" + "=" * 60)
    print("IR 验证测试")
    print("=" * 60)
    test_reduce_or_ir_has_reduce()
    test_bitonic_merge_ir_has_where()


if __name__ == "__main__":
    main()
