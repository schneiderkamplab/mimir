from typing import Optional

import torch
from torch import Tensor

from mimir.model.flash_attention_prefixlm_common import PREFIXLM_ROUTING_KEYS, prefixlm_routing_from_tensors

__all__ = ["flash_attn_varlen_prefixlm"]


def _rocm_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> Tensor:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "accelerator_type=rocm requires the ROCm FlashAttention 2 build exposing flash_attn.flash_attn_varlen_func."
        ) from exc

    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )


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
) -> Tensor:
    routing_values = (
        prefix_cu_seqlens,
        prefix_idx,
        causal_idx,
        active_key_idx,
        active_cu_seqlens_q,
        active_cu_seqlens_k,
    )
    if all(value is None for value in routing_values):
        routing = prefixlm_routing_from_tensors(
            prefix_lens,
            causal_lens,
            cu_seqlens,
            total_seqlen,
            numseqs,
            max_seqlen_prefix,
            max_seqlen_causal,
            max_seqlen_all,
        )
    elif any(value is None for value in routing_values):
        raise ValueError(f"PrefixLM routing tensors must provide all of {PREFIXLM_ROUTING_KEYS}")
    else:
        routing = dict(
            zip(PREFIXLM_ROUTING_KEYS, routing_values, strict=True)
        )  # type: ignore[arg-type]

    max_prefix = int(max_seqlen_prefix.item())

    out = torch.zeros_like(q)
    prefix_idx = routing["prefix_idx"]
    out_bidir = _rocm_varlen(
        q[prefix_idx],
        k[prefix_idx],
        v[prefix_idx],
        cu_seqlens_q=routing["prefix_cu_seqlens"],
        cu_seqlens_k=routing["prefix_cu_seqlens"],
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        causal=is_causal,
    )
    out[prefix_idx] = out_bidir

    causal_idx = routing["causal_idx"]
    if causal_idx.numel() > 0:
        out_causal = _rocm_varlen(
            q[causal_idx],
            k[routing["active_key_idx"]],
            v[routing["active_key_idx"]],
            cu_seqlens_q=routing["active_cu_seqlens_q"],
            cu_seqlens_k=routing["active_cu_seqlens_k"],
            # FlashAttention accepts upper bounds here. Reuse the packed-batch
            # metadata instead of synchronizing two device reductions to Python.
            max_seqlen_q=int(max_seqlen_causal.item()),
            max_seqlen_k=int(max_seqlen_all.item()),
            causal=True,
        )
        out[causal_idx] = out_causal

    return out
