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
from unittest.mock import MagicMock
from triton.language.semantic import TritonSemantic


@pytest.fixture(scope="function")
def restore_npu_hf32_setting():
    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = True
        yield
    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32


def torch_dot_None(x0, x1):
    res = torch.matmul(x0, x1)
    return res


@triton.jit
def triton_dot_2_None(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_allow_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, allow_tf32=True)
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_input_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="tf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_ignore_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


testlist1 = [
    (10, 13, 35, 39),
]

testlist2 = [(16, 32, 16)]

typelist = [
    'float32',
]


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_None[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_allow_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    expected_msg = "allow_tf32 is not supported as 'True'"
    try:
        triton_dot_2_allow_tf32[1, 1, 1](z, x, y, B, C, D)
    except Exception as e:
        if expected_msg in str(e):
            return
    pytest.fail(f"Expected exception containing '{expected_msg}', but no exception was raised")
    

@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_input_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_input_tf32[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_ignore_tf32(sigtype, B, C, D):
    input_type = "bfloat16"
    x = test_common.generate_tensor((B, C), input_type).npu()
    y = test_common.generate_tensor((C, D), input_type).npu()
    z = torch.zeros((B, D), dtype=torch.float32).npu()

    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = False
        z_ref = torch_dot_None(x.to(torch.float32), y.to(torch.float32)).to(torch.float32)

    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32

    triton_dot_2_ignore_tf32[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


# ==================== 4D and 5D dot test cases ====================
# Triton 3.6 supports higher dimensional inputs for tl.dot
# The last two dimensions are contracted, and the preceding dimensions are treated as batch

@triton.jit
def triton_dot_4d(output_ptr, x_ptr, y_ptr, 
                  A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    """4D tensor dot: contract last two dims"""
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (aidx[:, None, None] < A) & (bidx[None, :, None] < B) & (cidx[None, None, :] < C)
    y_mask = (aidx[:, None, None] < A) & (cidx[None, :, None] < C) & (didx[None, None, :] < D)
    out_mask = (aidx[:, None, None] < A) & (bidx[None, :, None] < B) & (didx[None, None, :] < D)
    
    # Compute linear indices for 4D tensors
    Xidx = aidx[:, None, None] * B * C + bidx[None, :, None] * C + cidx[None, None, :]
    Yidx = aidx[:, None, None] * C * D + cidx[None, :, None] * D + didx[None, None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    
    ret = tl.dot(X, Y, input_precision="hf32")
    
    oidx = aidx[:, None, None] * B * D + bidx[None, :, None] * D + didx[None, None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_5d(output_ptr, x_ptr, y_ptr, 
                  A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, 
                  D: tl.constexpr, E: tl.constexpr):
    """5D tensor dot: contract last two dims"""
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    eidx = tl.arange(0, E)

    x_mask = (aidx[:, None, None, None] < A) & (bidx[None, :, None, None] < B) & \
             (cidx[None, None, :, None] < C) & (didx[None, None, None, :] < D)
    y_mask = (aidx[:, None, None, None] < A) & (bidx[None, :, None, None] < B) & \
             (didx[None, None, :, None] < D) & (eidx[None, None, None, :] < E)
    out_mask = (aidx[:, None, None, None] < A) & (bidx[None, :, None, None] < B) & \
               (cidx[None, None, :, None] < C) & (eidx[None, None, None, :] < E)
    
    # Compute linear indices for 5D tensors
    Xidx = aidx[:, None, None, None] * B * C * D + bidx[None, :, None, None] * C * D + \
           cidx[None, None, :, None] * D + didx[None, None, None, :]
    Yidx = aidx[:, None, None, None] * B * D * E + bidx[None, :, None, None] * D * E + \
           didx[None, None, :, None] * E + eidx[None, None, None, :]
    
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    
    ret = tl.dot(X, Y, input_precision="hf32")
    
    oidx = aidx[:, None, None, None] * B * C * E + bidx[None, :, None, None] * C * E + \
           cidx[None, None, :, None] * E + eidx[None, None, None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


# 4D test parameters
testlist_4d = [
    (2, 8, 16, 12),   # A=2, B=8, C=16, D=12
    (4, 4, 32, 16),   # A=4, B=4, C=32, D=16
]

# 5D test parameters  
testlist_5d = [
    (2, 2, 8, 16, 12),   # A=2, B=2, C=8, D=16, E=12
    (2, 4, 4, 32, 16),   # A=2, B=4, C=4, D=32, E=16
]


@pytest.mark.parametrize("A, B, C, D", testlist_4d)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_4d(restore_npu_hf32_setting, sigtype, A, B, C, D):
    """Test 4D tensor dot operation"""
    x = test_common.generate_tensor((A, B, C), sigtype).npu()
    y = test_common.generate_tensor((A, C, D), sigtype).npu()
    z = torch.zeros((A, B, D), dtype=torch.float32).npu()
    
    # Reference: bmm over last two dims for each batch
    z_ref = torch.zeros((A, B, D), dtype=torch.float32).npu()
    for a in range(A):
        z_ref[a] = torch.matmul(x[a], y[a])
    
    triton_dot_4d[1, 1, 1](z, x, y, A, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.parametrize("A, B, C, D, E", testlist_5d)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_5d(restore_npu_hf32_setting, sigtype, A, B, C, D, E):
    """Test 5D tensor dot operation"""
    x = test_common.generate_tensor((A, B, C, D), sigtype).npu()
    y = test_common.generate_tensor((A, B, D, E), sigtype).npu()
    z = torch.zeros((A, B, C, E), dtype=torch.float32).npu()
    
    # Reference: bmm over last two dims for each batch
    z_ref = torch.zeros((A, B, C, E), dtype=torch.float32).npu()
    for a in range(A):
        for b in range(B):
            z_ref[a, b] = torch.matmul(x[a, b], y[a, b])
    
    triton_dot_5d[1, 1, 1](z, x, y, A, B, C, D, E)
    test_common.validate_cmp(sigtype, z, z_ref)


# =============================================================================
# Unit tests for the dot monkey-patch (third_party/ascend/backend/__init__.py)
# =============================================================================


def _make_mock_builder():
    builder = MagicMock()
    builder.options.default_dot_input_precision = "ieee"
    builder.options.allowed_dot_input_precisions = ("ieee", "hf32")
    builder.options.max_num_imprecise_acc_default = 0
    builder.codegen_fns = {"min_dot_size": MagicMock(return_value=(1, 1, 1))}
    builder.create_dot = MagicMock(return_value=MagicMock())
    builder.create_splat = MagicMock(return_value=MagicMock())
    builder.get_fp32 = MagicMock(return_value=MagicMock())
    builder.get_fp16 = MagicMock(return_value=MagicMock())
    builder.get_fp64 = MagicMock(return_value=MagicMock())
    builder.get_int32 = MagicMock(return_value=MagicMock())
    return builder


def _mock_tensor(dtype):
    t = MagicMock()
    t.dtype = dtype
    t.handle = MagicMock()
    dim = MagicMock()
    dim.value = 64
    t.shape = [dim, dim]
    type_mock = MagicMock()
    type_mock.is_block = MagicMock(return_value=True)
    type_mock.scalar = dtype
    type_mock.shape = [64, 64]
    t.type = type_mock
    return t


class TestDotAscendPatch:
    """Verify the monkey-patched TritonSemantic.dot is behaviourally
    equivalent to the original intrusive code."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        if getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
            return
        try:
            from triton.backends.ascend import _apply_ascend_patch as _apply
        except ImportError:
            pytest.skip("Ascend backend unavailable")
        _apply()

    @pytest.fixture
    def semantic(self):
        semantic = MagicMock(spec=TritonSemantic)
        semantic.builder = _make_mock_builder()
        semantic._str_to_dot_input_precision = (lambda ip: TritonSemantic._str_to_dot_input_precision(semantic, ip))
        semantic.tensor = MagicMock(return_value=MagicMock())
        semantic.cast = MagicMock(side_effect=lambda x, dtype: x)
        semantic.bitcast = MagicMock(side_effect=lambda x, dtype: x)
        return semantic

    # ---- HF32 guard ----

    def test_hf32_fp32_inputs_keep_hf32(self, semantic):
        """fp32 x fp32 + hf32 → hf32 is preserved."""
        patched = TritonSemantic.dot
        mock_original = MagicMock(return_value=MagicMock())
        cell = patched.__closure__[0]
        saved = cell.cell_contents
        cell.cell_contents = mock_original
        try:
            lhs = _mock_tensor(tl.float32)
            rhs = _mock_tensor(tl.float32)
            patched(semantic, lhs, rhs, None, "hf32", None, tl.float32)
            assert mock_original.call_args[0][4] == "hf32"
        finally:
            cell.cell_contents = saved

    def test_hf32_fp16_inputs_fallback_to_ieee(self, semantic):
        """fp16 x fp16 + hf32 → falls back to ieee."""
        patched = TritonSemantic.dot
        mock_original = MagicMock(return_value=MagicMock())
        cell = patched.__closure__[0]
        saved = cell.cell_contents
        cell.cell_contents = mock_original
        try:
            lhs = _mock_tensor(tl.float16)
            rhs = _mock_tensor(tl.float16)
            patched(semantic, lhs, rhs, None, "hf32", None, tl.float32)
            assert mock_original.call_args[0][4] == "ieee"
        finally:
            cell.cell_contents = saved

    # ---- max_num_imprecise_acc ----

    def test_max_imprecise_explicit_forced_to_zero(self, semantic, capsys):
        """Explicit max_num_imprecise_acc: warns, reaches create_dot as 0."""
        lhs = _mock_tensor(tl.float16)
        rhs = _mock_tensor(tl.float16)
        TritonSemantic.dot(semantic, lhs, rhs, None, "ieee", 16, tl.float32)

        assert "max_num_imprecise_acc" in capsys.readouterr().out
        assert semantic.builder.create_dot.call_args[0][4] == 0

    def test_max_imprecise_none_no_warning_and_zero(self, semantic, capsys):
        """None (default): no warning, reaches create_dot as 0."""
        lhs = _mock_tensor(tl.float16)
        rhs = _mock_tensor(tl.float16)
        TritonSemantic.dot(semantic, lhs, rhs, None, "ieee", None, tl.float32)

        assert "max_num_imprecise_acc" not in capsys.readouterr().out
        assert semantic.builder.create_dot.call_args[0][4] == 0
