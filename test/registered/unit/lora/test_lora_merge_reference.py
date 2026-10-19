"""The comparison reference must preserve factors and score before forcing."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def load_script(name):
    path = Path(__file__).resolve().parents[4] / "benchmark/lora" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMergeReference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base, self.adapter = self.root / "base", self.root / "adapter"
        self.base.mkdir()
        self.adapter.mkdir()
        self.merger = load_script("merge_reference")
        self.config = {"r": 2, "lora_alpha": 4, "bias": "none"}

    def fixture(self, base, factors):
        save_file(base, str(self.base / "model-1.safetensors"))
        index = {"weight_map": {k: "model-1.safetensors" for k in base}}
        (self.base / "model.safetensors.index.json").write_text(json.dumps(index))
        (self.base / "config.json").write_text('{"test":true}')
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.config))
        save_file(factors, str(self.adapter / "adapter_model.safetensors"))

    def test_plain_merge_is_fp32_then_bf16_and_source_is_untouched(self):
        w = torch.arange(12).reshape(4, 3).to(torch.bfloat16) / 17
        a = torch.tensor(
            [[0.13, -0.29, 0.43], [0.91, -0.21, 0.11]], dtype=torch.bfloat16
        )
        b = torch.arange(8).reshape(4, 2).to(torch.bfloat16) / 13
        self.fixture(
            {"model.proj.weight": w},
            {
                "base_model.model.model.proj.lora_A.weight": a,
                "base_model.model.model.proj.lora_B.weight": b,
            },
        )
        source_hash = self.merger.sha256(self.base / "model-1.safetensors")
        self.merger.merge(self.base, self.adapter, self.root / "merged", "cpu")
        result = load_file(str(self.root / "merged/model-1.safetensors"))[
            "model.proj.weight"
        ]
        self.assertTrue(
            torch.equal(result, (w.float() + 2 * (b.float() @ a.float())).bfloat16())
        )
        self.assertEqual(
            source_hash, self.merger.sha256(self.base / "model-1.safetensors")
        )
        manifest = json.loads((self.root / "merged/merge-manifest.json").read_text())
        self.assertEqual(manifest["tensors"][0]["shape"], [4, 3])
        self.assertGreater(manifest["tensors"][0]["rounding_abs_max"], 0)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.merger.merge(self.base, self.adapter, self.root / "merged", "cpu")

    def test_fused_qkvz_preserves_both_output_updates(self):
        a = torch.arange(6).reshape(2, 3).float() / 7
        b = torch.arange(12).reshape(6, 2).float() / 13
        qkv, z = torch.ones(4, 3).bfloat16(), torch.ones(2, 3).bfloat16()
        self.fixture(
            {
                "model.linear_attn.in_proj_qkv.weight": qkv,
                "model.linear_attn.in_proj_z.weight": z,
            },
            {
                "base_model.model.model.linear_attn.in_proj_qkvz.lora_A.weight": a,
                "base_model.model.model.linear_attn.in_proj_qkvz.lora_B.weight": b,
            },
        )
        self.merger.merge(self.base, self.adapter, self.root / "merged", "cpu")
        merged = load_file(str(self.root / "merged/model-1.safetensors"))
        self.assertTrue(
            torch.equal(
                merged["model.linear_attn.in_proj_qkv.weight"],
                (qkv.float() + 2 * b[:4] @ a).bfloat16(),
            )
        )
        self.assertTrue(
            torch.equal(
                merged["model.linear_attn.in_proj_z.weight"],
                (z.float() + 2 * b[4:] @ a).bfloat16(),
            )
        )

    def test_fp32_export_does_not_round_updated_weights_to_bf16(self):
        w = torch.ones(2, 3, dtype=torch.bfloat16)
        a = torch.full((2, 3), 0.01, dtype=torch.bfloat16)
        b = torch.full((2, 2), 0.01, dtype=torch.bfloat16)
        self.fixture(
            {"model.proj.weight": w},
            {
                "base_model.model.model.proj.lora_A.weight": a,
                "base_model.model.model.proj.lora_B.weight": b,
            },
        )
        self.merger.merge(
            self.base, self.adapter, self.root / "merged", "cpu", "float32"
        )
        merged = load_file(str(self.root / "merged/model-1.safetensors"))[
            "model.proj.weight"
        ]
        self.assertEqual(merged.dtype, torch.float32)
        self.assertTrue(torch.equal(merged, w.float() + 2 * (b.float() @ a.float())))
        self.assertFalse(torch.equal(merged, merged.bfloat16().float()))

    def test_unsupported_peft_options_fail_before_creating_output(self):
        self.config["use_rslora"] = True
        self.fixture({}, {})
        with self.assertRaisesRegex(ValueError, "use_rslora"):
            self.merger.merge(self.base, self.adapter, self.root / "merged", "cpu")
        self.assertFalse((self.root / "merged").exists())

    def test_capture_scores_original_logits_before_forcing_decode(self):
        comparator = load_script("compare_merged")
        original = torch.tensor([[1.0, 2.0, 3.0]])
        logits = original.clone()
        capture = self.root / "capture.jsonl"
        params = {
            "__req__": SimpleNamespace(output_ids=[0]),
            "_capture_step": 1,
            "continuation": [0, 0],
            "capture_file": str(capture),
        }
        comparator.CaptureAndForce()(logits, [params])
        row = json.loads(capture.read_text())
        self.assertEqual(row["phase"], "decode")
        self.assertEqual(row["step"], 1)
        self.assertAlmostEqual(row["logprob"], original.log_softmax(-1)[0, 0].item())
        self.assertEqual(row["argmax"], 2)
        self.assertEqual(logits.argmax(-1).item(), 0)
        self.assertEqual(logits[0, 0].item(), 0)
        self.assertTrue(torch.isneginf(logits[0, 1:]).all())

    def test_capture_step_advances_even_when_output_commit_lags(self):
        comparator = load_script("compare_merged")
        capture = self.root / "overlap-capture.jsonl"
        params = {
            "__req__": SimpleNamespace(output_ids=[]),
            "continuation": [0, 1, 2],
            "capture_file": str(capture),
        }
        processor = comparator.CaptureAndForce()
        for target in (0, 1, 2):
            logits = torch.tensor([[1.0, 2.0, 3.0]])
            processor(logits, [params])
            self.assertEqual(logits.argmax(-1).item(), target)
        rows = [json.loads(x) for x in capture.read_text().splitlines()]
        self.assertEqual([x["step"] for x in rows], [0, 1, 2])
        self.assertEqual(
            [x["phase"] for x in rows], ["prefill_last", "decode", "decode"]
        )


if __name__ == "__main__":
    unittest.main()
