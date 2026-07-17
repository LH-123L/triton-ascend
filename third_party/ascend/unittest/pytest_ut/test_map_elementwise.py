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


# ==================== Triton 3.6 map_elementwise tests ====================
# Triton 3.6 added tl.map_elementwise for element-wise scalar function mapping

@triton.jit
def identity_scalar(x):
    return x


@triton.jit
def add_one_scalar(x):
    return x + 1.0


@triton.jit
def square_scalar(x):
    return x * x


@triton.jit
def add_scalar(x, y):
    return x + y


@triton.jit
def relu_scalar(x):
    if x > 0:
        return x
    else:
        return 0.0


@triton.jit
def divmod_scalar(a, b):
    return a // b, a % b


@triton.jit
def map_elementwise_identity_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    out = tl.map_elementwise(identity_scalar, x)
    tl.store(out_ptr + offs, out)


@triton.jit
def map_elementwise_add_one_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    out = tl.map_elementwise(add_one_scalar, x)
    tl.store(out_ptr + offs, out)


@triton.jit
def map_elementwise_square_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    out = tl.map_elementwise(square_scalar, x)
    tl.store(out_ptr + offs, out)


@triton.jit
def map_elementwise_add_kernel(in_ptr_a, in_ptr_b, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(in_ptr_a + offs)
    b = tl.load(in_ptr_b + offs)
    out = tl.map_elementwise(add_scalar, a, b)
    tl.store(out_ptr + offs, out)


@triton.jit
def map_elementwise_relu_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    out = tl.map_elementwise(relu_scalar, x)
    tl.store(out_ptr + offs, out)


@triton.jit
def map_elementwise_divmod_kernel(in_ptr_a, in_ptr_b, out_ptr_q, out_ptr_r, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(in_ptr_a + offs)
    b = tl.load(in_ptr_b + offs)
    q, r = tl.map_elementwise(divmod_scalar, a, b)
    tl.store(out_ptr_q + offs, q)
    tl.store(out_ptr_r + offs, r)


@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_map_elementwise_identity(N, dtype):
    """Test map_elementwise with identity function"""
    torch.manual_seed(42)
    x = torch.randn(N, dtype=dtype, device="npu")
    out = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_identity_kernel[(1,)](x, out, N=N)
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_map_elementwise_add_one(N, dtype):
    """Test map_elementwise with add one function"""
    torch.manual_seed(42)
    x = torch.randn(N, dtype=dtype, device="npu")
    out = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_add_one_kernel[(1,)](x, out, N=N)
    expected = x + 1.0
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_map_elementwise_square(N, dtype):
    """Test map_elementwise with square function"""
    torch.manual_seed(42)
    x = torch.randn(N, dtype=dtype, device="npu")
    out = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_square_kernel[(1,)](x, out, N=N)
    expected = x * x
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_map_elementwise_add(N, dtype):
    """Test map_elementwise with multi-argument add function"""
    torch.manual_seed(42)
    a = torch.randn(N, dtype=dtype, device="npu")
    b = torch.randn(N, dtype=dtype, device="npu")
    out = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_add_kernel[(1,)](a, b, out, N=N)
    expected = a + b
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_map_elementwise_relu(N, dtype):
    """Test map_elementwise with relu function (branching)"""
    torch.manual_seed(42)
    x = torch.randn(N, dtype=dtype, device="npu")
    out = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_relu_kernel[(1,)](x, out, N=N)
    expected = torch.clamp(x, min=0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_map_elementwise_divmod(N, dtype):
    """Test map_elementwise with multi-output divmod function"""
    torch.manual_seed(42)
    a = torch.randint(1, 100, (N,), dtype=dtype, device="npu")
    b = torch.randint(1, 10, (N,), dtype=dtype, device="npu")
    out_q = torch.empty(N, dtype=dtype, device="npu")
    out_r = torch.empty(N, dtype=dtype, device="npu")
    
    map_elementwise_divmod_kernel[(1,)](a, b, out_q, out_r, N=N)
    expected_q = a // b
    expected_r = a % b
    
    torch.testing.assert_close(out_q, expected_q)
    torch.testing.assert_close(out_r, expected_r)