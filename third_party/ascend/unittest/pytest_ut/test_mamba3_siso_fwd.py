import torch

from mamba_ssm.ops.triton.mamba3 import mamba3_siso_fwd

def main():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    device = "npu"
    batch, seqlen, nheads_qk, headdim_qk = 1, 4096, 1, 128
    nheads = 80
    headdim_angles = 32
    headdim_v = 64
    Q = torch.randn(batch, seqlen, nheads_qk, headdim_qk, dtype=torch.float32, device=device)
    K = torch.randn(batch, seqlen, nheads_qk, headdim_qk, dtype=torch.float32, device=device)
    V = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, device=device)
    ADT = torch.randn(batch, nheads, seqlen, dtype=torch.float32, device=device)
    DT = torch.randn(batch, nheads, seqlen, dtype=torch.float32, device=device)
    Trap = torch.randn(batch, nheads, seqlen, dtype=torch.float32, device=device)
    Q_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, device=device)
    K_bias = torch.randn(nheads, headdim_qk, dtype=torch.float32, device=device)
    Angles_Cumsum = torch.randn(batch, seqlen, nheads, headdim_angles, dtype=torch.float32, device=device)
    D = torch.randn(nheads, dtype=torch.float32, device=device)
    Z = torch.randn(batch, seqlen, nheads, headdim_v, dtype=torch.float32, device=device)
    Input_States = None
    is_outproj_norm = False
    z = None
    chunk_size = 16
    ssm_state = None

    Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma, Final_States = mamba3_siso_fwd(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum, D, Z, Input_States,
        chunk_size=chunk_size,
        store_states_adt_outv=False,
        return_final_states=False,
        cu_seqlens=None,
    )

main()
