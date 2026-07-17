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
import pytest
import math
import test_common


@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_histogram(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}')).npu()
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
def test_histogram_uint(M, N, ncore, dtype):
    torch.manual_seed(17)
    x_cpu = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}'), device="cpu")
    x = x_cpu.to("npu")
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.{dtype}'))
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)


# ==================== Triton 3.6 mask parameter support ====================
# Triton 3.6 added mask parameter to tl.histogram

@triton.jit
def histogram_mask_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    # Create a mask that excludes certain elements
    mask = x >= 0    # All elements included
    z = tl.histogram(x, N, mask=mask)
    tl.store(z_ptr + offset2, z)


@triton.jit
def histogram_selective_mask_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, threshold: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    # Only count elements below threshold
    mask = x < threshold
    z = tl.histogram(x, N, mask=mask)
    tl.store(z_ptr + offset2, z)


@triton.jit
def histogram_partial_mask_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    # Mask out specific range
    mask = (x >= 0) & (x < N // 2)
    z = tl.histogram(x, N, mask=mask)
    tl.store(z_ptr + offset2, z)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("dtype", ["int32"])
def test_histogram_with_mask(M, N, dtype):
    """Test histogram with mask parameter (Triton 3.6 feature)"""
    torch.manual_seed(42)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()
    
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_mask_kernel[(1,)](x, y_ref, M=M, N=N)
    
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("threshold", [4])
@pytest.mark.parametrize("dtype", ["int32"])
def test_histogram_selective_mask(M, N, threshold, dtype):
    """Test histogram with selective mask (Triton 3.6 feature)"""
    torch.manual_seed(42)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()
    
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_selective_mask_kernel[(1,)](x, y_ref, M=M, N=N, threshold=threshold)
    
    # Reference: only count elements < threshold
    masked_x = x[x < threshold]
    y_cal = torch.histc(masked_x.float(), bins=N, min=0, max=N - 1)
    y_cal = torch.where(torch.arange(N, device="npu") >= threshold, torch.tensor(0, device="npu"), y_cal)
    
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("dtype", ["int32"])
def test_histogram_partial_mask(M, N, dtype):
    """Test histogram with partial mask (Triton 3.6 feature)"""
    torch.manual_seed(42)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()
    
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_partial_mask_kernel[(1,)](x, y_ref, M=M, N=N)
    
    # Reference: only count elements in [0, N//2)
    mask_range = (x >= 0) & (x < N // 2)
    masked_x = x[mask_range]
    y_cal = torch.histc(masked_x.float(), bins=N, min=0, max=N - 1)
    
    test_common.validate_cmp(dtype, y_cal, y_ref)
