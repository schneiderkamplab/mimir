# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts; also runnable with unittest without the GPU test harness."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Mimir2Config

from vllm.model_executor.models.mimir2 import (
    Mimir2ForCausalLM,
    Mimir2ServingNGram,
    pack_weights,
    recurrence_steps,
    validate_config,
)
from vllm.model_executor.models.mimir2_routing import (
    prepare_ngram_windows,
    prepare_sparse_mos_inputs,
)


class Mimir2Contracts(unittest.TestCase):
    def test_ngram_windows_and_compiled_math(self):
        from transformers.models.mimir2.modeling_mimir2 import Mimir2NGramMemory

        config = Mimir2Config(
            hidden_size=16,
            num_attention_heads=2,
            ngram_orders=[2, 3],
            ngram_buckets=31,
            ngram_heads=4,
            ngram_contextual_gate=True,
        )
        native = Mimir2NGramMemory(config)
        serving = Mimir2ServingNGram(config)
        with torch.no_grad():
            native.alpha.fill_(0.7)
            for table in native.tables:
                table.normal_(std=0.1)
        serving.load_state_dict(native.state_dict(), strict=True)
        compiled = torch.compile(serving, backend="aot_eager", fullgraph=True)
        records = np.array([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]])
        embeddings = torch.randn(2, 6, 16)
        full = native(embeddings, torch.tensor(records), torch.arange(6))
        # Unequal chunk lengths, decode, request reorder and slot replacement.
        for order, starts, lengths in [
            ([0, 1], [0, 0], [3, 2]),
            ([1, 0], [2, 3], [1, 2]),
            ([0, 1], [5, 3], [1, 3]),
        ]:
            batch = SimpleNamespace(
                num_reqs=2,
                token_ids_cpu=records[order],
                is_token_ids=np.ones((2, 6), dtype=bool),
                num_computed_tokens_cpu=np.array(starts),
            )
            bounds = np.array([0, lengths[0], sum(lengths)])
            windows = prepare_ngram_windows(batch, bounds, sum(lengths) + 1, [2, 3])
            x = torch.cat(
                [
                    embeddings[row, start : start + length]
                    for row, start, length in zip(order, starts, lengths)
                ]
                + [torch.zeros(1, 16)]
            )
            expected = torch.cat(
                [
                    full[row, start : start + length]
                    for row, start, length in zip(order, starts, lengths)
                ]
                + [torch.zeros(1, 16)]
            )
            torch.testing.assert_close(serving(x, windows), expected)
            torch.testing.assert_close(compiled(x, windows), expected)
        batch.num_computed_tokens_cpu[:] = 0
        batch.token_ids_cpu[:] = 7
        windows = prepare_ngram_windows(batch, [0, 1, 2], 2, [2, 3])
        self.assertEqual(windows.tolist(), [[7, -1, -1], [7, -1, -1]])

    def test_request_owned_routes(self):
        batch = SimpleNamespace(
            num_reqs=2,
            num_prompt_tokens=np.array([2, 3]),
            token_ids_cpu=np.array([[11, 12, 99], [21, 22, 23]]),
            is_token_ids=np.ones((2, 3), dtype=bool),
        )
        inputs = prepare_sparse_mos_inputs(batch, [0, 1, 3], 4, "cpu", "prefix_mean")
        self.assertEqual(inputs["mos_prompt_ids"].tolist(), [11, 12, 21, 22, 23])
        self.assertEqual(inputs["mos_query_owners"].tolist(), [0, 1, 1, -1])
        batch.token_ids_cpu = batch.token_ids_cpu[::-1].copy()
        batch.num_prompt_tokens = batch.num_prompt_tokens[::-1].copy()
        inputs = prepare_sparse_mos_inputs(batch, [0, 1, 2], 2, "cpu", "first_token")
        self.assertEqual(inputs["mos_prompt_ids"].tolist(), [21, 11])
        batch.token_ids_cpu[0, 0] = 42
        inputs = prepare_sparse_mos_inputs(batch, [0, 1, 2], 2, "cpu", "first_token")
        self.assertEqual(inputs["mos_prompt_ids"].tolist(), [42, 11])

    def test_runner_stages_stable_ngram_buffer(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        config = Mimir2Config(ngram_orders=[2, 3])
        batch = SimpleNamespace(
            num_reqs=2,
            num_computed_tokens_cpu=np.array([0, 1]),
            token_ids_cpu=np.array([[1, 2, 3], [9, 8, 7]]),
            is_token_ids=np.ones((2, 3), dtype=bool),
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            max_num_tokens=8,
            device="cpu",
            is_pooling_model=False,
            input_batch=batch,
            query_start_loc=SimpleNamespace(np=np.array([0, 2, 3])),
            get_model=lambda: SimpleNamespace(),
            model=lambda **kwargs: kwargs["ngram_windows"],
        )
        GPUModelRunner._init_model_kwargs(runner)
        address = runner.mimir2_ngram_windows.data_ptr()
        GPUModelRunner._init_model_kwargs(runner)
        self.assertEqual(address, runner.mimir2_ngram_windows.data_ptr())
        windows = GPUModelRunner._model_forward(
            runner,
            input_ids=torch.tensor([1, 2, 8, 0]),
            positions=torch.tensor([0, 1, 1, 0]),
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        self.assertEqual(address, windows.data_ptr())
        self.assertEqual(
            windows.tolist(), [[1, -1, -1], [2, 1, -1], [8, 9, -1], [-1, -1, -1]]
        )
        batch.num_computed_tokens_cpu[:] = [2, 2]
        runner.query_start_loc.np = np.array([0, 1, 2])
        windows = GPUModelRunner._model_forward(
            runner,
            input_ids=torch.tensor([3, 7]),
            positions=torch.tensor([2, 2]),
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        self.assertEqual(address, windows.data_ptr())
        self.assertEqual(windows.tolist(), [[3, 2, 1], [7, 8, 9]])

    def test_ngram_strict_loader(self):
        model = Mimir2ForCausalLM.__new__(Mimir2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = Mimir2Config(
            hidden_size=16,
            num_attention_heads=2,
            tie_word_embeddings=False,
            ngram_orders=[2, 3],
            ngram_buckets=31,
            ngram_heads=4,
            ngram_contextual_gate=True,
        )
        model.model = torch.nn.Module()
        model.model.ngram = Mimir2ServingNGram(model.config)
        model.lm_head = torch.nn.Linear(16, 32, bias=False)
        weights = [(name, value.clone()) for name, value in model.state_dict().items()]
        self.assertEqual(model.load_weights(weights), set(model.state_dict()))
        for missing in (
            "model.ngram.query.weight",
            "model.ngram.key.weight",
            "model.ngram.tables.7",
            "model.ngram.alpha",
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                model.load_weights(
                    [(key, value) for key, value in weights if key != missing]
                )

    def test_loader_requires_feature_buffers(self):
        model = Mimir2ForCausalLM.__new__(Mimir2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.model = torch.nn.Module()
        model.model.register_buffer("gate_alpha", torch.tensor(0.0))
        model.model.z_L_init = torch.nn.Parameter(torch.zeros(4))
        model.lm_head = torch.nn.Linear(4, 8, bias=False)
        weights = [
            (name, value.clone() + 0.5) for name, value in model.state_dict().items()
        ]
        loaded = model.load_weights(weights)
        self.assertEqual(loaded, set(model.state_dict()))
        self.assertEqual(model.model.gate_alpha.item(), 0.5)
        with self.assertRaisesRegex(ValueError, "gate_alpha"):
            model.load_weights([(k, v) for k, v in weights if k != "model.gate_alpha"])

    def test_cache_slots(self):
        for experts in (1, 2, 4):
            config = Mimir2Config(mos_num_experts=experts)
            slots = recurrence_steps(config, "H")
            for expert in range(experts):
                slots += recurrence_steps(config, "L", expert)
            self.assertEqual(sorted(slots), list(range(len(slots))))
            self.assertEqual(
                len(slots) * config.num_layers_per_stack, config.num_hidden_layers
            )

    def test_packing(self):
        for prefix, projections, packed in (
            ("model.L_module.layers.0.self_attn", ("gate", "q", "k", "v"), "gqkv"),
            ("model.L_module.layers.0.mlp", ("gate", "up"), "gate_up"),
            (
                "model.L_module.stack_experts.0.layers.0.mlp.experts.1",
                ("gate", "up"),
                "gate_up",
            ),
        ):
            tensors = [torch.full((2, 3), float(i)) for i in range(len(projections))]
            weights = list(
                zip([f"{prefix}.{p}_proj.weight" for p in projections], tensors)
            )
            result = dict(pack_weights(reversed(weights)))
            torch.testing.assert_close(
                result[f"{prefix}.{packed}_proj.weight"], torch.cat(tensors)
            )
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                list(pack_weights(weights[:-1]))
            with self.assertRaisesRegex(ValueError, "Mixed"):
                list(
                    pack_weights(
                        weights + [(f"{prefix}.{packed}_proj.weight", tensors[0])]
                    )
                )

    def test_duplicate(self):
        weights = [("model.gate_alpha", torch.tensor(0.5))] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            list(pack_weights(weights))

    def test_unsupported_features(self):
        for fields in ({"ngram_orders": [2]},):
            config = SimpleNamespace(
                model_config=SimpleNamespace(hf_config=Mimir2Config(**fields)),
                parallel_config=SimpleNamespace(pipeline_parallel_size=1),
                scheduler_config=SimpleNamespace(async_scheduling=True),
                quant_config=None,
                speculative_config=None,
            )
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
