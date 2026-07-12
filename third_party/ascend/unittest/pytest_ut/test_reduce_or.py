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

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

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
    """reduce_or 1D basic logical or reduction"""
    N = 16
    x = torch.zeros(N, dtype=torch.int32, device='npu')
    x[3] = 1
    x[10] = 1
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    expected = (x != 0).any().to(torch.int32).reshape(1)
    assert out.item() == expected.item(), f"reduce_or error: {out} vs {expected}"
    print("PASS: test_reduce_or_basic")


def test_reduce_or_all_zero():
    """reduce_or all zeros should return 0"""
    N = 16
    x = torch.zeros(N, dtype=torch.int32, device='npu')
    out = torch.ones(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    assert out.item() == 0, f"all-zero reduce_or should be 0, got {out}"
    print("PASS: test_reduce_or_all_zero")


def test_reduce_or_all_one():
    """reduce_or all ones should return 1"""
    N = 16
    x = torch.ones(N, dtype=torch.int32, device='npu')
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1,)](x, out, N=N)
    assert out.item() == 1, f"all-one reduce_or should be 1, got {out}"
    print("PASS: test_reduce_or_all_one")


def test_reduce_or_2d():
    """reduce_or 2D along axis=1"""
    M, N = 4, 8
    x = torch.zeros(M, N, dtype=torch.int32, device='npu')
    x[0, 2] = 1
    x[1, 5] = 1
    x[2, :] = 0  # all zeros
    x[3, 0] = 1
    out = torch.zeros(M, dtype=torch.int32, device='npu')
    reduce_or_2d_kernel[(1,)](x, out, M=M, N=N)
    expected = (x != 0).any(dim=1).to(torch.int32)
    assert torch.equal(out, expected), f"reduce_or 2D error: {out} vs {expected}"
    print("PASS: test_reduce_or_2d")


def main():
    test_reduce_or_basic()
    test_reduce_or_all_zero()
    test_reduce_or_all_one()
    test_reduce_or_2d()
