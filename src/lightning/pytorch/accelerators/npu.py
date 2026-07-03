from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException

class NPUAccelerator(Accelerator):
    """Accelerator for Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not NPU.
        """
        if device.type != "npu":
            raise MisconfigurationException(f"Device should be NPU, got {device} instead.")
        torch.npu.set_device(device)

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        return torch.npu.memory_stats(device)

    @override
    def teardown(self) -> None:
        torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic.

        -1 or '-1' means use all npus.

        """

        if isinstance(devices, list):
            return devices
        if isinstance(devices, str):
            if devices == "-1":
                return list(range(torch.npu.device_count()))
            if "," in devices:
                return [int(x.strip()) for x in devices.split(",") if len(x) > 0]
            return list(range(int(devices.strip())))
        if isinstance(devices, int):
            if devices == -1:
                return list(range(torch.npu.device_count()))
            return list(range(devices))

        return None

    @staticmethod
    @override
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""

        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""

        return torch.npu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        try:
            import torch_npu  # noqa: F401

            return torch.npu.device_count() > 0
        except ImportError:
            # NPU may raise these exceptions if it's not properly configured.
            return False

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @override
    def get_distribute_name(self) -> str:
        return "hccl"

    @override
    def get_stream_context(self, device_id: List[int]) -> Any:
        return torch.npu.stream(torch.npu.Stream()) if device_id is not None else nullcontext()

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "npu",
            cls,
            description=cls.__name__,
        )
