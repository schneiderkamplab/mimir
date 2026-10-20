from typing import Optional

import torch
from torch import Tensor

from mimir.model.accelerator import get_accelerator_type

__all__ = ["flash_attn_varlen_prefixlm"]


def _mps_kernel_supported(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return q.device.type == "mps" and q.dtype == torch.float32 and q.shape == k.shape == v.shape


def _mps_prefixlm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    from mimir.model.flash_attention_prefixlm_mps import flash_attn_varlen_prefixlm_mps

    return flash_attn_varlen_prefixlm_mps(
        q,
        k,
        v,
        is_causal,
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )


def _dense_prefixlm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
) -> Tensor:
    from mimir.model.flash_attention_prefixlm_dense import flash_attn_varlen_prefixlm as dense_prefixlm

    return dense_prefixlm(
        q,
        k,
        v,
        is_causal,
        prefix_lens,
        causal_lens,
        cu_seqlens,
        total_seqlen,
        numseqs,
        max_seqlen_prefix,
        max_seqlen_causal,
        max_seqlen_all,
    )


@torch.compiler.disable
def flash_attn_varlen_prefixlm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
    *,
    prefix_cu_seqlens: Optional[Tensor] = None,
    prefix_idx: Optional[Tensor] = None,
    causal_idx: Optional[Tensor] = None,
    active_key_idx: Optional[Tensor] = None,
    active_cu_seqlens_q: Optional[Tensor] = None,
    active_cu_seqlens_k: Optional[Tensor] = None,
    cu_seqlens_shifted: Optional[Tensor] = None,
    prefix_mask: Optional[Tensor] = None,
    causal_mask: Optional[Tensor] = None,
    fa4_impl: str = "seqused",
    fa4_grad_mask_impl: str = "triton",
    fa4_output_combine_impl: str = "triton",
) -> Tensor:
    match get_accelerator_type():
        case "sm90":
            raise RuntimeError("SM90 PrefixLM attention is implemented in mimir.model.flash_attention_prefixlm_v2.")
        case "sm100":
            from mimir.model.flash_attention_prefixlm_fa4 import flash_attn_varlen_prefixlm as fa4_prefixlm

            return fa4_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
                prefix_cu_seqlens=prefix_cu_seqlens,
                prefix_idx=prefix_idx,
                causal_idx=causal_idx,
                active_key_idx=active_key_idx,
                active_cu_seqlens_q=active_cu_seqlens_q,
                active_cu_seqlens_k=active_cu_seqlens_k,
                cu_seqlens_shifted=cu_seqlens_shifted,
                prefix_mask=prefix_mask,
                causal_mask=causal_mask,
                impl=fa4_impl,
                grad_mask_impl=fa4_grad_mask_impl,
                output_combine_impl=fa4_output_combine_impl,
            )
        case "rocm":
            from mimir.model.flash_attention_prefixlm_rocm import flash_attn_varlen_prefixlm as rocm_prefixlm

            return rocm_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
                prefix_cu_seqlens=prefix_cu_seqlens,
                prefix_idx=prefix_idx,
                causal_idx=causal_idx,
                active_key_idx=active_key_idx,
                active_cu_seqlens_q=active_cu_seqlens_q,
                active_cu_seqlens_k=active_cu_seqlens_k,
            )
        case "mps":
            if _mps_kernel_supported(q, k, v):
                return _mps_prefixlm(
                    q,
                    k,
                    v,
                    is_causal,
                    prefix_lens,
                    causal_lens,
                    cu_seqlens,
                    total_seqlen,
                    numseqs,
                    max_seqlen_prefix,
                    max_seqlen_causal,
                    max_seqlen_all,
                )
            return _dense_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
        case "cpu" | "none":
            return _dense_prefixlm(
                q,
                k,
                v,
                is_causal,
                prefix_lens,
                causal_lens,
                cu_seqlens,
                total_seqlen,
                numseqs,
                max_seqlen_prefix,
                max_seqlen_causal,
                max_seqlen_all,
            )
    raise ValueError(f"Unsupported accelerator_type: {get_accelerator_type()}")
