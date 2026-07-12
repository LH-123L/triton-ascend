# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu

import triton
import triton.language as tl

import test_common


# ============================================================
# Triton kernels
# ============================================================
@triton.jit
def bitonic_merge_kernel(in_ptr, out_ptr, N: tl.constexpr,
                         DESCENDING: tl.constexpr):
    """Bitonic merge on a 1D tensor of length N (power of two)."""
    x = tl.load(in_ptr + tl.arange(0, N))
    y = tl.bitonic_merge(x, descending=DESCENDING)
    tl.store(out_ptr + tl.arange(0, N), y)


@triton.jit
def bitonic_merge_2d_kernel(in_ptr, out_ptr, M: tl.constexpr,
                            N: tl.constexpr, DESCENDING: tl.constexpr):
    """Bitonic merge on a 2D tensor along the last dimension."""
    x = tl.load(in_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    y = tl.bitonic_merge(x, descending=DESCENDING)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], y)


def _make_bitonic_1d(N, dtype, device='npu', low=0, high=1000):
    """
    Build a 1D bitonic sequence of length N by concatenating an ascending
    half and a descending half. A bitonic sequence is the legal input for
    bitonic_merge and the output is guaranteed to be fully sorted.
    """
    half = N // 2
    if dtype.is_floating_point:
        a = torch.sort(torch.randn(half, dtype=dtype, device=device)).values
        b = torch.sort(torch.randn(half, dtype=dtype, device=device),
                       descending=True).values
    else:
        a = torch.sort(torch.randint(low, high, (half,),
                                     dtype=dtype, device=device)).values
        b = torch.sort(torch.randint(low, high, (half,),
                                     dtype=dtype, device=device),
                       descending=True).values
    return torch.cat([a, b])


def _make_bitonic_2d(M, N, dtype, device='npu', low=0, high=1000):
    """
    Build a 2D tensor of shape (M, N) where each row is a bitonic sequence
    along the last dimension.
    """
    rows = [_make_bitonic_1d(N, dtype, device, low, high) for _ in range(M)]
    return torch.stack(rows, dim=0)


# ============================================================
# Test cases: ascending merge
# ============================================================
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64])
def test_bitonic_merge_ascending_fp32(N):
    """Bitonic merge on float32 bitonic input, ascending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.float32)
    out = torch.empty(N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_bitonic_merge_ascending_fp16(N):
    """Bitonic merge on float16 bitonic input, ascending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.float16)
    out = torch.empty(N, dtype=torch.float16, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('float16', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_bitonic_merge_ascending_bf16(N):
    """Bitonic merge on bfloat16 bitonic input, ascending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.bfloat16)
    out = torch.empty(N, dtype=torch.bfloat16, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('bfloat16', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_bitonic_merge_ascending_int32(N):
    """Bitonic merge on int32 bitonic input, ascending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.int32)
    out = torch.empty(N, dtype=torch.int32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('int32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_bitonic_merge_ascending_int64(N):
    """Bitonic merge on int64 bitonic input, ascending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.int64)
    out = torch.empty(N, dtype=torch.int64, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('int64', out.cpu(), expected.cpu())


# ============================================================
# Test cases: descending merge
# ============================================================
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64])
def test_bitonic_merge_descending_fp32(N):
    """Bitonic merge on float32 bitonic input, descending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.float32)
    out = torch.empty(N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=1)
    expected, _ = torch.sort(x, descending=True)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_bitonic_merge_descending_int32(N):
    """Bitonic merge on int32 bitonic input, descending order."""
    torch.manual_seed(0)
    x = _make_bitonic_1d(N, torch.int32)
    out = torch.empty(N, dtype=torch.int32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=1)
    expected, _ = torch.sort(x, descending=True)
    test_common.validate_cmp('int32', out.cpu(), expected.cpu())


# ============================================================
# Test cases: 2D tensor (multi-row merge along last dim)
# ============================================================
@pytest.mark.parametrize("M,N", [(2, 4), (4, 8), (8, 16), (4, 32)])
def test_bitonic_merge_2d_fp32(M, N):
    """Bitonic merge on 2D float32 bitonic rows along last dim."""
    torch.manual_seed(0)
    x = _make_bitonic_2d(M, N, torch.float32)
    out = torch.empty(M, N, dtype=torch.float32, device='npu')
    bitonic_merge_2d_kernel[(1,)](x, out, M=M, N=N, DESCENDING=0)
    expected, _ = torch.sort(x, dim=-1)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("M,N", [(2, 4), (4, 8), (8, 16)])
def test_bitonic_merge_2d_int32(M, N):
    """Bitonic merge on 2D int32 bitonic rows along last dim."""
    torch.manual_seed(0)
    x = _make_bitonic_2d(M, N, torch.int32)
    out = torch.empty(M, N, dtype=torch.int32, device='npu')
    bitonic_merge_2d_kernel[(1,)](x, out, M=M, N=N, DESCENDING=0)
    expected, _ = torch.sort(x, dim=-1)
    test_common.validate_cmp('int32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("M,N", [(4, 8), (8, 16)])
def test_bitonic_merge_2d_descending_fp32(M, N):
    """Bitonic merge on 2D float32 bitonic rows along last dim, descending."""
    torch.manual_seed(0)
    x = _make_bitonic_2d(M, N, torch.float32)
    out = torch.empty(M, N, dtype=torch.float32, device='npu')
    bitonic_merge_2d_kernel[(1,)](x, out, M=M, N=N, DESCENDING=1)
    expected, _ = torch.sort(x, dim=-1, descending=True)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


# ============================================================
# Test cases: bitonic input pattern (two sorted halves concatenated)
# ============================================================
@pytest.mark.parametrize("N", [8, 16, 32])
def test_bitonic_merge_concat_sorted_halves_ascending(N):
    """
    Concatenate two sorted halves (asc + desc) to form a bitonic sequence,
    then merge into ascending order.
    """
    torch.manual_seed(0)
    half = N // 2
    a = torch.sort(torch.randn(half, dtype=torch.float32, device='npu')).values
    b = torch.sort(torch.randn(half, dtype=torch.float32, device='npu'), descending=True).values
    x = torch.cat([a, b])
    out = torch.empty(N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


@pytest.mark.parametrize("N", [8, 16, 32])
def test_bitonic_merge_concat_sorted_halves_descending(N):
    """
    Concatenate two sorted halves (desc + asc) to form a bitonic sequence,
    then merge into descending order.
    """
    torch.manual_seed(0)
    half = N // 2
    a = torch.sort(torch.randn(half, dtype=torch.float32, device='npu'), descending=True).values
    b = torch.sort(torch.randn(half, dtype=torch.float32, device='npu')).values
    x = torch.cat([a, b])
    out = torch.empty(N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=N, DESCENDING=1)
    expected, _ = torch.sort(x, descending=True)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


# ============================================================
# Test cases: boundary cases
# ============================================================
def test_bitonic_merge_min_size():
    """Bitonic merge on minimum power-of-two size: N=2."""
    x = torch.tensor([3.0, 1.0], dtype=torch.float32, device='npu')
    out = torch.empty(2, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=2, DESCENDING=0)
    expected = torch.tensor([1.0, 3.0], dtype=torch.float32)
    test_common.validate_cmp('float32', out.cpu(), expected)


def test_bitonic_merge_already_sorted_ascending():
    """Bitonic merge on an already ascending tensor (idempotency check)."""
    x = torch.arange(16, dtype=torch.float32, device='npu')
    out = torch.empty(16, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=16, DESCENDING=0)
    test_common.validate_cmp('float32', out.cpu(), x.cpu())


def test_bitonic_merge_already_sorted_descending():
    """Bitonic merge on an already descending tensor (idempotency check)."""
    x = torch.arange(15, -1, -1, dtype=torch.float32, device='npu')
    out = torch.empty(16, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=16, DESCENDING=1)
    test_common.validate_cmp('float32', out.cpu(), x.cpu())


def test_bitonic_merge_all_equal():
    """Bitonic merge on a tensor where all elements are equal."""
    x = torch.full((16,), 5.0, dtype=torch.float32, device='npu')
    out = torch.empty(16, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=16, DESCENDING=0)
    test_common.validate_cmp('float32', out.cpu(), x.cpu())


def test_bitonic_merge_with_negatives():
    """Bitonic merge with negative values present in a bitonic sequence."""
    torch.manual_seed(0)
    # Build a bitonic sequence from a larger range that includes negatives.
    half = 16
    a = torch.sort(torch.randn(half, dtype=torch.float32, device='npu') * 10).values
    b = torch.sort(torch.randn(half, dtype=torch.float32, device='npu') * 10,
                   descending=True).values
    x = torch.cat([a, b])  # bitonic sequence of length 32
    out = torch.empty(32, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=32, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('float32', out.cpu(), expected.cpu())


def test_bitonic_merge_with_duplicates():
    """Bitonic merge on a bitonic sequence that contains duplicate values."""
    # Build a bitonic sequence with duplicates:
    #   ascending half: [1, 2, 2, 5]
    #   descending half: [8, 8, 5, 1]
    #   concatenated: [1, 2, 2, 5, 8, 8, 5, 1]  (bitonic)
    x = torch.tensor([1, 2, 2, 5, 8, 8, 5, 1], dtype=torch.int32, device='npu')
    out = torch.empty(8, dtype=torch.int32, device='npu')
    bitonic_merge_kernel[(1,)](x, out, N=8, DESCENDING=0)
    expected, _ = torch.sort(x)
    test_common.validate_cmp('int32', out.cpu(), expected.cpu())
