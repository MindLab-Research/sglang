import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmark" / "lora"))
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
from compare_frozen import CaptureAndForce, digest, main, read_scores
from pd_native import NativePDClient


class TestFrozenClient(unittest.TestCase):
    def test_original_scores_and_decode_start(self):
        with tempfile.TemporaryDirectory() as d:
            params = {
                "capture_file": str(Path(d) / "scores"),
                "capture_start_step": 1,
                "continuation": [0, 2, 1],
            }
            x = torch.tensor([[1.0, 2.0, 3.0]])
            expected = x.log_softmax(-1)[0, 2].item()
            with patch("torch.distributed.get_rank", return_value=3):
                CaptureAndForce()(x, [params])
            row = json.loads(Path(params["capture_file"] + ".rank-3").read_text())
            self.assertEqual(
                (row["step"], row["token"], row["logprob"], row["argmax"]),
                (1, 2, expected, 2),
            )
            self.assertEqual(x.argmax(-1).item(), 2)

    def test_rank_disagreement_or_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scores"
            row = [
                {"step": i, "token": v, "logprob": -0.5, "argmax": v}
                for i, v in enumerate([2, 1])
            ]
            for rank in [0, 1]:
                Path(str(p) + f".rank-{rank}").write_text(
                    "".join(json.dumps(r) + "\n" for r in row)
                )
            res = {
                "output_ids": [2, 1],
                "meta_info": {
                    "input_token_logprobs": [[None, 3], [-1.0, 4]],
                    "input_top_logprobs": [None, [[-1.0, 4]]],
                },
            }
            scores = read_scores(res, p, {"input_ids": [3, 4]}, [2, 1], 2)
            self.assertEqual(len(scores["decode"]), 1)
            row[1]["logprob"] = -0.6
            Path(str(p) + ".rank-1").write_text(
                "".join(json.dumps(r) + "\n" for r in row)
            )
            with self.assertRaises(AssertionError):
                read_scores(res, p, {"input_ids": [3, 4]}, [2, 1], 2)
            Path(str(p) + ".rank-1").unlink()
            with self.assertRaises(AssertionError):
                read_scores(res, p, {"input_ids": [3, 4]}, [2, 1], 2)

    def test_pd_pair_uses_same_room_and_preserves_prefill_scores(self):
        with tempfile.TemporaryDirectory() as d:
            calls = {}
            original = {
                "input_ids": [[3, 4]],
                "custom_logit_processor": ["serialized"],
                "sampling_params": [
                    {
                        "custom_params": {
                            "capture_file": "shared",
                            "continuation": [2, 1],
                        }
                    }
                ],
            }
            saved = copy.deepcopy(original)

            def request(url, path, payload):
                calls[url] = payload
                return [
                    {
                        "output_ids": [2] if url == "p" else [2, 1],
                        "meta_info": {
                            "input_token_logprobs": (
                                [[None, 3], [-1.0, 4]] if url == "p" else []
                            )
                        },
                    }
                ]

            with patch("pd_native.call", side_effect=request):
                res = NativePDClient("p", "d", 8998, d).call("d", "/generate", original)
            self.assertEqual(original, saved)
            self.assertEqual(calls["p"]["bootstrap_room"], calls["d"]["bootstrap_room"])
            self.assertIsInstance(calls["p"]["bootstrap_room"][0], int)
            self.assertEqual(
                calls["p"]["sampling_params"][0]["custom_params"]["capture_start_step"],
                0,
            )
            self.assertEqual(
                calls["d"]["sampling_params"][0]["custom_params"]["capture_start_step"],
                1,
            )
            self.assertEqual(
                res[0]["meta_info"]["input_token_logprobs"], [[None, 3], [-1.0, 4]]
            )

    def test_pd_capture_joins_prefill_and_decode_without_losing_a_step(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scores"
            prefill = {"step": 0, "token": 2, "logprob": -0.25, "argmax": 2}
            decode = {"step": 1, "token": 1, "logprob": -0.5, "argmax": 1}
            for suffix, row in (("prefill", prefill), ("decode", decode)):
                for rank in range(2):
                    Path(f"{path}.{suffix}.rank-{rank}").write_text(
                        json.dumps(row) + "\n"
                    )
            response = {
                "output_ids": [2, 1],
                "meta_info": {
                    "input_token_logprobs": [[None, 3], [-1.0, 4]],
                    "input_top_logprobs": [None, [[-1.0, 4]]],
                },
            }
            scores = read_scores(
                response, path, {"input_ids": [3, 4]}, [2, 1], 2, pd=True
            )
            self.assertEqual(scores["prefill_last"], [(-0.25, 2)])
            self.assertEqual(scores["decode"], [(-0.5, 1)])
            for rank in range(2):
                decode["step"] = 0
                Path(f"{path}.decode.rank-{rank}").write_text(json.dumps(decode) + "\n")
            with self.assertRaises(AssertionError):
                read_scores(response, path, {"input_ids": [3, 4]}, [2, 1], 2, pd=True)

    def test_tampered_reference_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "record").write_text("original")
            (r / "seal.json").write_text(
                json.dumps({"files": {"record": digest(r / "record")}})
            )
            (r / "record").write_text("changed")
            with patch(
                "compare_frozen.call",
                side_effect=AssertionError("network must not be reached"),
            ) as network:
                with self.assertRaises(AssertionError):
                    main(SimpleNamespace(stage="dynamic", reference=r))
                network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
