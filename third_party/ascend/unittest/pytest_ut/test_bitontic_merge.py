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
def bitonic_merge_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, N))
    b = tl.load(b_ptr + tl.arange(0, N))
    # bitonic_merge accepts two sorted sequences and outputs the merged sorted sequence
    merged = tl.bitonic_merge(a, b, order=True)
    tl.store(out_ptr + tl.arange(0, 2 * N), merged)


def test_bitonic_merge_basic():
    """bitonic_merge merging two ascending sequences"""
    N = 8
    a = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.float32, device='npu')
    b = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32, device='npu')
    out = torch.empty(2 * N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](a, b, out, N=N)
    expected = torch.cat([a, b]).sort().values
    assert torch.allclose(out, expected), f"bitonic_merge error: {out} vs {expected}"
    print("PASS: test_bitonic_merge_basic")


def test_bitonic_merge_descending():
    """bitonic_merge merging two descending sequences"""
    N = 4
    a = torch.tensor([15, 11, 7, 3], dtype=torch.float32, device='npu')
    b = torch.tensor([14, 10, 6, 2], dtype=torch.float32, device='npu')
    out = torch.empty(2 * N, dtype=torch.float32, device='npu')
    bitonic_merge_kernel[(1,)](a, b, out, N=N)
    expected = torch.cat([a, b]).sort(descending=True).values
    assert torch.allclose(out, expected), f"bitonic_merge descending error: {out} vs {expected}"
    print("PASS: test_bitonic_merge_descending")