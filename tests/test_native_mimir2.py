# Copyright 2026 The Mimir authors and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Self-contained native/Transformers Mimir2 parity."""

import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from transformers.models.mimir2.convert_mimir2_weights_to_hf import convert_checkpoint


class Mimir2NativeParityTest(unittest.TestCase):
    def test_native_features(self):
        from mimir.model.mimir2 import Mimir2
        from mimir.model.lm_head import LMHead

        def attention(q, k, v, causal, **kwargs):
            mask = None
            if not causal:
                positions = torch.arange(q.shape[-3])
                mask = (positions[:, None] >= positions[None, :]) | (
                    (positions[:, None] < 2) & (positions[None, :] < 2)
                )
            return F.scaled_dot_product_attention(
                q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2), is_causal=causal, attn_mask=mask
            ).transpose(-3, -2)

        features = [
            {"moe_num_experts": 3, "moe_top_k": 2, "moe_aux_loss_free": True},
            {"mos_num_experts": 3, "mos_top_k": 1, "mos_aux_loss_free": True},
            {"moe_num_experts": 3, "moe_top_k": 1, "moe_aux_loss_free": True, "moe_sparse": True,
             "mos_num_experts": 2, "mos_top_k": 1, "mos_aux_loss_free": True},
            {},
            {"gates_enabled": True},
            {"moe_num_experts": 3, "moe_top_k": 2, "moe_sparse": True},
            {"mos_num_experts": 2, "mos_top_k": 2, "mos_state_mode": "separate"},
            {"residual_branches": 2, "residual_input_dependent": True},
            {"h_memory_branches": 2, "l_memory_branches": 2},
            {"ngram_orders": [2, 3]},
            {"ngram_orders": [2, 3], "ngram_heads": 4, "ngram_contextual_gate": True},
            {
                "gates_enabled": True,
                "moe_num_experts": 2,
                "moe_top_k": 2,
                "moe_levels": "both",
                "mos_num_experts": 2,
                "mos_top_k": 2,
                "residual_branches": 2,
                "h_memory_branches": 2,
                "l_memory_branches": 2,
                "ngram_orders": [2, 3],
            },
        ]
        for extra in features:
            with self.subTest(extra=extra), patch("mimir.model.layers.flash_attn_varlen_prefixlm", attention):
                config = dict(
                    hidden_size=16,
                    num_heads=2,
                    n_layers=2,
                    half_layers=True,
                    expansion=1,
                    norm_type="pre",
                    norm_eps=1e-6,
                    init_type="lecun_normal",
                    pos_emb_type="rope",
                    rope_theta=10000,
                    max_seq_len=16,
                    H_cycles=2,
                    L_cycles=3,
                    bp_max_steps=8,
                    bp_warmup_ratio=0,
                    fwd_bwd_dtype="float32",
                    vocab_size=32,
                    attn_type="causal",
                    **extra,
                )
                torch.manual_seed(123)
                native = LMHead(Mimir2(config), config).eval()
                with torch.no_grad():
                    for name, param in native.named_parameters():
                        if any(
                            part in name for part in ("invocation_gates", "read_gain", "ngram.tables", "router.rows")
                        ):
                            param.normal_(std=0.1)
                    for name, buffer in native.named_buffers():
                        if name.endswith("selection_bias"):
                            buffer.copy_(torch.linspace(-.2,.2,buffer.numel()))
                        if name.endswith("alpha"):
                            buffer.fill_(0.7)
                converted = convert_checkpoint(native.state_dict(), config).eval()
                ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
                positions = torch.arange(ids.shape[1]).expand_as(ids)
                with torch.no_grad():
                    expected = native(None, {"inputs": ids, "position_ids": positions}, bp_steps=8)[1]
                    actual = converted(ids, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                if extra.get("moe_aux_loss_free") or extra.get("mos_aux_loss_free"):
                    with torch.no_grad():
                        prefill=converted(ids[:,:-1],use_cache=True)
                        decode=converted(ids[:,-1:],past_key_values=prefill.past_key_values,use_cache=True)
                    torch.testing.assert_close(decode.logits[:,-1],actual[:,-1],rtol=2e-5,atol=2e-6)
                if extra.get("ngram_contextual_gate"):
                    # Loaded-weight native greedy reference versus HF cached generation.
                    tokens = ids.clone()
                    with torch.no_grad():
                        for _ in range(3):
                            logits = native(None, {"inputs": tokens,
                                "position_ids": torch.arange(tokens.shape[1]).expand_as(tokens)}, bp_steps=8)[1]
                            tokens = torch.cat((tokens, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
                        generated = converted.generate(ids, max_new_tokens=3, do_sample=False,
                                                       eos_token_id=None, pad_token_id=0)
                    torch.testing.assert_close(tokens, generated, rtol=0, atol=0)
                    # Missing contextual weights must fail strict export.
                    broken = dict(native.state_dict())
                    del broken["model.ngram.query.weight"]
                    with self.assertRaises(RuntimeError):
                        convert_checkpoint(broken, config)
                for module in native.modules():
                    if hasattr(module, "attn_type"):
                        module.attn_type = "prefixlm"
                converted.config.prefix_lm = True
                prefix = torch.zeros_like(ids)
                prefix[:, :2] = 1
                with torch.no_grad():
                    expected = native(None, {"inputs": ids, "position_ids": positions}, bp_steps=8)[1]
                    actual = converted(ids, token_type_ids=prefix, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                for module in native.modules():
                    if hasattr(module, "attn_type"):
                        module.attn_type = "causal"
                native = native.to(torch.bfloat16)
                converted = convert_checkpoint(native.state_dict(), config).eval()
                with torch.no_grad():
                    expected = native(None, {"inputs": ids, "position_ids": positions}, bp_steps=8)[1]
                    actual = converted(ids, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.03)

    def test_ngram_packed_and_interleaved_requests(self):
        from mimir.model.ngram_memory import NGramMemory
        from transformers import Mimir2Config
        from transformers.models.mimir2.modeling_mimir2 import Mimir2NGramMemory
        native = NGramMemory(16, [2, 3], 31, 4, True)
        hf = Mimir2NGramMemory(Mimir2Config(hidden_size=16, num_attention_heads=2,
            ngram_orders=[2, 3], ngram_buckets=31, ngram_heads=4, ngram_contextual_gate=True))
        with torch.no_grad():
            native.alpha.fill_(.7)
            for table in native.tables:
                table.normal_(std=.1)
        hf.load_state_dict(native.state_dict(), strict=True)
        x = torch.randn(2, 8, 16)
        ids = torch.randint(0, 32, (2, 8))
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 0, 1, 2, 3]])
        expected = native(x, ids, positions)
        torch.testing.assert_close(hf(x, ids, positions), expected, rtol=0, atol=0)
        histories = [{}, {}]
        for start, stop in ((0, 2), (2, 5), (5, 8)):
            for row in (1, 0):
                actual = hf(x[row:row+1, start:stop], ids[row:row+1, start:stop],
                            positions[row:row+1, start:stop], histories[row])
                torch.testing.assert_close(actual, expected[row:row+1, start:stop])
