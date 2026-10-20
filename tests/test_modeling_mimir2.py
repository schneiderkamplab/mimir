# Copyright 2026 The Mimir authors and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import tempfile
import unittest

import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    HrmTextConfig,
    HrmTextForCausalLM,
    Mimir2Config,
    Mimir2ForCausalLM,
    Mimir2Model,
)
from transformers.models.mimir2.modeling_mimir2 import Mimir2DynamicCache


def tiny_config(**kwargs):
    return Mimir2Config(
        **(
            {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
                "eos_token_id": None,
                "embedding_scale": 1.0,
                "bp_steps": 8,
            }
            | kwargs
        )
    )


class Mimir2ModelTest(unittest.TestCase):
    def test_auto_roundtrip(self):
        config = tiny_config(gates_enabled=True, mos_num_experts=2)
        model = AutoModelForCausalLM.from_config(config).eval()
        ids = torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored_config = AutoConfig.from_pretrained(folder)
            self.assertEqual(restored_config.num_layers_per_stack, 1)
            self.assertEqual(restored_config.num_hidden_layers, 14)
            restored = AutoModelForCausalLM.from_pretrained(folder).eval()
            torch.testing.assert_close(model(ids).logits, restored(ids).logits)
            self.assertIsInstance(AutoModel.from_config(restored_config), Mimir2Model)

    def test_v1_default_parity(self):
        config = tiny_config()
        v1_config = HrmTextConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
        baseline = HrmTextForCausalLM(v1_config).eval()
        model = Mimir2ForCausalLM(config).eval()
        model.load_state_dict(baseline.state_dict(), strict=True)
        ids = torch.tensor([[1, 2, 3, 4]])
        prefix = torch.tensor([[1, 1, 0, 0]])
        with torch.no_grad():
            torch.testing.assert_close(
                baseline(ids, token_type_ids=prefix, use_cache=False).logits,
                model(ids, token_type_ids=prefix, use_cache=False).logits,
                rtol=0,
                atol=0,
            )

    def test_cached_features_and_prefix(self):
        for features in (
            {},
            {"gates_enabled": True},
            {"moe_num_experts": 3, "moe_top_k": 2, "moe_sparse": True},
            {"mos_num_experts": 2, "mos_top_k": 2, "mos_state_mode": "separate"},
            {"residual_branches": 2, "residual_input_dependent": True},
            {"h_memory_branches": 2, "l_memory_branches": 2},
            {"ngram_orders": [2, 3]},
            {"ngram_orders": [2, 3], "ngram_heads": 4, "ngram_contextual_gate": True},
        ):
            with self.subTest(features=features):
                model = Mimir2ForCausalLM(tiny_config(**features)).eval()
                # Nonzero features catch accidental no-op implementations.
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if any(part in name for part in ("invocation_gates", "read_gain", "ngram.tables")):
                            param.normal_(std=0.1)
                    for name, buffer in model.named_buffers():
                        if name.endswith("alpha"):
                            buffer.fill_(0.7)
                    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
                    prefix = torch.tensor([[1, 1, 0, 0, 0]]).expand_as(ids)
                    full = model(ids, token_type_ids=prefix, use_cache=False).logits
                    prefill = model(ids[:, :3], token_type_ids=prefix[:, :3], use_cache=True)
                    cached = model(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True).logits
                    torch.testing.assert_close(full[:, 3:], cached, rtol=1e-4, atol=1e-5)

    def test_generate_and_beams(self):
        for features in ({}, {"ngram_orders": [2, 3], "ngram_heads": 4, "ngram_contextual_gate": True, "ngram_alpha": 1.0}):
            model = Mimir2ForCausalLM(tiny_config(**features)).eval()
            ids = torch.tensor([[1, 2, 3]])
            prefix = torch.ones_like(ids)
            with torch.no_grad():
                a = model.generate(ids, token_type_ids=prefix, max_new_tokens=3, do_sample=False)
                b = model.generate(ids, token_type_ids=prefix, max_new_tokens=3, do_sample=False, use_cache=False)
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(model.generate(ids, max_new_tokens=3, num_beams=2).shape, (1, 6))

    def test_gradient_boundaries(self):
        for bp in (2, 5, 8):
            model = Mimir2ForCausalLM(tiny_config(gates_enabled=True, gate_alpha=1.0, bp_steps=bp))
            model(torch.tensor([[1, 2, 3]]), use_cache=False).logits.square().mean().backward()
            h_count = min(2, bp - 1)
            for level, active in ((model.model.H_module, h_count), (model.model.L_module, bp - h_count)):
                for index, gate in enumerate(level.invocation_gates):
                    self.assertEqual(gate.weight.grad is not None, index >= len(level.invocation_gates) - active)

    def test_sparse_mos_full_context(self):
        model = Mimir2ForCausalLM(tiny_config(mos_num_experts=2, mos_top_k=2, mos_sparse=True)).eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(model(ids, use_cache=False).logits.shape, (2, 3, 32))
        self.assertIsNotNone(model(ids, use_cache=True).past_key_values)

    def test_sparse_cached_routes_and_padding(self):
        for policy in ("first_token", "prefix_mean"):
            for state in ("shared", "separate"):
                for active in (1, 2):
                    with self.subTest(policy=policy, state=state, active=active):
                        torch.manual_seed(42)
                        model = Mimir2ForCausalLM(tiny_config(
                            mos_num_experts=3, mos_top_k=active, mos_sparse=True,
                            mos_state_mode=state, mos_sparse_routing=policy,
                            gates_enabled=True, gate_alpha=0.7,
                        )).eval()
                        with torch.no_grad():
                            for row in model.model.L_module.stack_router.rows:
                                row.normal_(std=0.5)
                            ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
                            mask = (ids != 0).long()
                            types = torch.tensor([[0, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
                            full = model(ids, attention_mask=mask, token_type_ids=types, use_cache=False).logits
                            prefill = model(ids[:, :3], attention_mask=mask[:, :3], token_type_ids=types[:, :3])
                            cache = prefill.past_key_values
                            routes = cache.mos_routes.clone()
                            decoded = model(ids[:, 3:], attention_mask=mask, past_key_values=cache).logits
                            torch.testing.assert_close(decoded, full[:, 3:], atol=1e-5, rtol=1e-4)
                            torch.testing.assert_close(cache.mos_routes, routes)
                            self.assertEqual(cache.get_seq_length(), 5)
                            cache.reorder_cache(torch.tensor([1, 0, 1]))
                            torch.testing.assert_close(cache.mos_routes, routes[[1, 0, 1]])
                            cache.batch_select_indices(torch.tensor([1, 0]))
                            cache.batch_repeat_interleave(2)
                            torch.testing.assert_close(cache.mos_routes, routes.repeat_interleave(2, 0))
                            with self.assertRaisesRegex(ValueError, "Cropping sparse MoS"):
                                cache.crop(-1)
                            cache.reset()
                            self.assertIsNone(cache.mos_routes)
                            self.assertEqual(cache.get_seq_length(), 0)

    def test_sparse_beam_generation_and_inactive_first_expert(self):
        model = Mimir2ForCausalLM(tiny_config(mos_num_experts=3, mos_top_k=1, mos_sparse=True)).eval()
        model.model.L_module.stack_router.logit_bias.copy_(torch.tensor([-10., -10., 10.]))
        calls = []
        handle = model.model.L_module.layers[0].register_forward_pre_hook(lambda *args: calls.append(1))
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            cached = model.generate(ids, token_type_ids=torch.ones_like(ids), num_beams=3, max_new_tokens=3)
            full = model.generate(ids, token_type_ids=torch.ones_like(ids), num_beams=3, max_new_tokens=3, use_cache=False)
        handle.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(cached, full))

    def test_ngram_cache_reordering(self):
        cache = Mimir2DynamicCache(config=tiny_config())
        cache.ngram_history["tokens"] = torch.tensor([[1, 2], [3, 4]])
        cache.reorder_cache(torch.tensor([1, 0]))
        self.assertEqual(cache.ngram_history["tokens"].tolist(), [[3, 4], [1, 2]])

    def test_all_features_roundtrip(self):
        model = Mimir2ForCausalLM(
            tiny_config(
                gates_enabled=True,
                moe_num_experts=2,
                moe_top_k=2,
                mos_num_experts=2,
                mos_top_k=2,
                residual_branches=2,
                residual_input_dependent=True,
                h_memory_branches=2,
                l_memory_branches=2,
                ngram_orders=[2, 3],
            )
        ).eval()
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name.endswith("alpha"):
                    buffer.fill_(0.6)
            for name, param in model.named_parameters():
                if "ngram.tables" in name or "invocation_gates" in name:
                    param.normal_(std=0.1)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = AutoModelForCausalLM.from_pretrained(folder).eval()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            ids = torch.tensor([[1, 2, 3, 4]])
            torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)

    def test_left_padding(self):
        model = Mimir2ForCausalLM(tiny_config()).eval()
        with torch.no_grad():
            plain = model(torch.tensor([[1, 2, 3]]), token_type_ids=torch.tensor([[1, 1, 0]])).logits
            padded = model(
                torch.tensor([[0, 0, 1, 2, 3]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
                token_type_ids=torch.tensor([[0, 0, 1, 1, 0]]),
            ).logits
        torch.testing.assert_close(plain, padded[:, -3:], rtol=1e-4, atol=1e-5)

    def test_gradient_checkpointing(self):
        for branches in (1, 2):
            config = tiny_config(residual_branches=branches, gates_enabled=True, gate_alpha=1.0)
            model = Mimir2ForCausalLM(config).train()
            reference = Mimir2ForCausalLM(config).train()
            reference.load_state_dict(model.state_dict())
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            ids = torch.tensor([[1, 2, 3]])
            a = model(ids, use_cache=False).logits
            b = reference(ids, use_cache=False).logits
            torch.testing.assert_close(a, b)
            a.square().mean().backward()
            b.square().mean().backward()
            for name, param in model.named_parameters():
                expected = dict(reference.named_parameters())[name].grad
                if expected is not None:
                    torch.testing.assert_close(param.grad, expected)

    def test_validation(self):
        for invalid in (
            {"moe_top_k": 2},
            {"mos_top_k": 0},
            {"ngram_orders": [1]},
            {"mos_state_mode": "separate", "l_memory_branches": 2},
            {"gate_alpha": 2.0},
        ):
            with self.assertRaises(ValueError):
                tiny_config(**invalid)
