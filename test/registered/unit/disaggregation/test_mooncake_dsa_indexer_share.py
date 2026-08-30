import concurrent.futures
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingEngine:
    def __init__(self):
        self.transfers = []

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.transfers.extend(zip(src_addrs, dst_addrs, lengths))
        return 0


class TestMooncakeDSAIndexerShare(unittest.TestCase):
    SRC_BASE = 1_000_000
    DST_BASE = 2_000_000
    LAYER_STRIDE = 10_000
    ITEM_LEN = 16

    @staticmethod
    def _indexer_types():
        # GLM-5.3: layers 0/1/2 own buffers, then one owner every four layers.
        return [
            "full" if layer_id < 3 or (layer_id - 2) % 4 == 0 else "shared"
            for layer_id in range(78)
        ]

    def _run_cp_slice(
        self,
        start_layer,
        end_layer,
        dcp_size,
        *,
        indexer_share=True,
        include_draft=False,
        state_type=StateType.DSA,
        custom_mem_pool=False,
    ):
        indexer_types = self._indexer_types()
        owner = None
        dst_ptrs = []
        for layer_id, indexer_type in enumerate(indexer_types):
            if indexer_type == "full":
                owner = layer_id
            self.assertIsNotNone(owner)
            dst_layer_id = owner if indexer_share else layer_id
            dst_ptrs.append(self.DST_BASE + dst_layer_id * self.LAYER_STRIDE)

        src_ptrs = [
            self.SRC_BASE + layer_id * self.LAYER_STRIDE
            for layer_id in range(start_layer, end_layer)
        ]
        if include_draft:
            src_ptrs.append(self.SRC_BASE + 999 * self.LAYER_STRIDE)
            dst_ptrs.append(self.DST_BASE + 999 * self.LAYER_STRIDE)
        item_lens = [self.ITEM_LEN] * len(src_ptrs)

        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = custom_mem_pool
        manager.attn_tp_size = dcp_size
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=start_layer,
            prefill_end_layer=end_layer - 1,
            mla_compression_ratios=None,
            state_types=[state_type],
            state_data_ptrs=[src_ptrs],
            state_item_lens=[item_lens],
            state_dim_per_tensor=[[]],
            state_conv_shard_groups=[[]],
        )
        manager._is_dsv4_kv_transfer = lambda: False
        manager.engine = _RecordingEngine()

        dcp_rank = 1
        virtual_dst_indices = np.array([dcp_rank, dcp_rank + dcp_size], dtype=np.int32)
        src_indices = np.array([100, 101], dtype=np.int32)
        req = TransferInfo(
            room=1,
            endpoint="127.0.0.1",
            dst_port=8998,
            mooncake_session_id="test-session",
            dst_kv_indices=np.array([], dtype=np.int32),
            dst_aux_index=0,
            dst_state_indices=[virtual_dst_indices.tolist()],
            required_dst_info_num=1,
            is_dummy=False,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        registration = KVArgsRegisterInfo(
            room="1",
            endpoint="127.0.0.1",
            dst_port=8998,
            mooncake_session_id="test-session",
            dst_kv_ptrs=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[dst_ptrs],
            dst_tp_rank=dcp_rank,
            dst_attn_tp_size=dcp_size,
            dst_kv_item_len=self.ITEM_LEN,
            dst_state_item_lens=[item_lens],
            dst_state_dim_per_tensor=[[]],
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            rc = manager.maybe_send_extra(
                req,
                [src_indices],
                executor,
                registration,
            )
        self.assertEqual(rc, 0)

        physical_dst_indices = virtual_dst_indices // dcp_size
        actual_transfers = [
            (
                (src_addr - src_indices[0] * self.ITEM_LEN - self.SRC_BASE)
                // self.LAYER_STRIDE,
                dst_addr - physical_dst_indices[0] * self.ITEM_LEN,
            )
            for src_addr, dst_addr, _ in manager.engine.transfers
        ]
        expected_owner_layers = list(range(start_layer, end_layer))
        if indexer_share and state_type == StateType.DSA:
            expected_owner_layers = [
                layer_id
                for layer_id in expected_owner_layers
                if indexer_types[layer_id] == "full"
            ]
        if include_draft:
            expected_owner_layers.append(999)

        self.assertEqual(
            sorted(actual_transfers),
            sorted(
                (
                    layer_id,
                    (
                        self.DST_BASE + 999 * self.LAYER_STRIDE
                        if layer_id == 999
                        else dst_ptrs[layer_id]
                    ),
                )
                for layer_id in expected_owner_layers
            ),
        )
        if indexer_share and state_type == StateType.DSA:
            transferred_dst_ptrs = [dst_ptr for _, dst_ptr in actual_transfers]
            self.assertEqual(len(transferred_dst_ptrs), len(set(transferred_dst_ptrs)))

    def test_alias_destinations_only_receive_owner_layers_for_dcp4_and_dcp8(self):
        # Cover a slice beginning at layer 0 and one beginning on a shared layer.
        for dcp_size in (4, 8):
            for start_layer, end_layer in ((0, 10), (20, 30)):
                with self.subTest(
                    dcp_size=dcp_size,
                    start_layer=start_layer,
                    end_layer=end_layer,
                ):
                    self._run_cp_slice(start_layer, end_layer, dcp_size)

    def test_full_model_layout_filters_aliases_and_preserves_draft_state(self):
        self._run_cp_slice(0, 78, 4)
        self._run_cp_slice(0, 78, 4, include_draft=True)

    def test_non_aliased_dsa_destinations_are_unchanged(self):
        self._run_cp_slice(20, 30, 4, indexer_share=False)

    def test_draft_state_after_aliased_target_layers_is_preserved(self):
        self._run_cp_slice(0, 10, 4, include_draft=True)

    def test_custom_mem_pool_path_filters_aliases(self):
        self._run_cp_slice(20, 30, 4, custom_mem_pool=True)

    def test_non_dsa_state_is_not_deduplicated(self):
        self._run_cp_slice(0, 10, 4, state_type=StateType.SWA)


if __name__ == "__main__":
    unittest.main()
