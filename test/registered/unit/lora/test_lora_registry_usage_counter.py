"""Unit tests for the LoRA registry usage counter that gates unload.

`LoRARegistry.acquire()` accounts one usage per dispatched request and
`unload_lora_adapter()` waits for the counter to reach zero before it frees the
adapter on the engines. These tests pin the two properties the engine relies on:
the counter is balanced by release(), and a usage that is never released keeps
the unload blocked (which is why every engine-side cleanup path must release).
"""

import asyncio
import unittest

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


async def _make_registry_and_usage():
    """Registry with one acquired (never released) usage, adapter unregistered."""
    registry = LoRARegistry()
    await registry.register(LoRARef(lora_name="a", lora_path="/tmp/a"))
    lora_id = await registry.acquire("a")
    # unload_lora_adapter() unregisters before it waits for the drain
    await registry.unregister("a")
    return registry, lora_id


class TestLoRAUsageCounter(CustomTestCase):
    def test_acquire_and_release_balance(self):
        async def drive():
            registry = LoRARegistry()
            await registry.register(LoRARef(lora_name="a", lora_path="/tmp/a"))
            lora_id = await registry.acquire("a")
            self.assertEqual(registry.pending_usage(lora_id), 1)
            await registry.release(lora_id)
            self.assertEqual(registry.pending_usage(lora_id), 0)

        asyncio.run(drive())

    def test_unload_wait_returns_once_usage_released(self):
        async def drive():
            registry, lora_id = await _make_registry_and_usage()

            async def release_later():
                await asyncio.sleep(0.05)
                await registry.release(lora_id)

            releaser = asyncio.create_task(release_later())
            await asyncio.wait_for(registry.wait_for_unload(lora_id), timeout=5)
            await releaser
            # wait_for_unload() drops the counter once the drain completed
            self.assertIsNone(registry.pending_usage(lora_id))

        asyncio.run(drive())

    def test_unreleased_usage_keeps_unload_blocked(self):
        """Documents the failure mode the tokenizer cleanup paths must prevent."""

        async def drive():
            registry, lora_id = await _make_registry_and_usage()

            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(registry.wait_for_unload(lora_id), timeout=0.2)
            self.assertEqual(registry.pending_usage(lora_id), 1)

        asyncio.run(drive())


if __name__ == "__main__":
    unittest.main()
