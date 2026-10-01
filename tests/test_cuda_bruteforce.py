from __future__ import annotations

import pytest

from stability import cuda_bruteforce


def test_cuda_stress_seed_varies_each_launch_and_stays_u32() -> None:
    first = cuda_bruteforce._stress_seed(0x13579BDF, 0, 0xA5A5A5A5)
    second = cuda_bruteforce._stress_seed(0x13579BDF, 1, 0xA5A5A5A5)

    assert first != second
    assert 0 <= first <= 0xFFFFFFFF
    assert 0 <= second <= 0xFFFFFFFF


def test_cuda_verification_samples_cover_edges_and_middle() -> None:
    element_count = cuda_bruteforce.DEFAULT_STRESS_ELEMENTS

    indices = cuda_bruteforce._verification_sample_indices(element_count)

    assert indices[:4] == [0, 1, 2, 7]
    assert element_count // 2 in indices
    assert element_count - 1 in indices
    assert len(indices) == len(set(indices))
    assert all(0 <= index < element_count for index in indices)


def test_cuda_cpu_reference_is_deterministic_for_stress_rounds() -> None:
    seed0 = cuda_bruteforce._stress_seed(0x13579BDF, 17, 0xA5A5A5A5)
    seed1 = cuda_bruteforce._stress_seed(0x2468ACE1, 17, 0x5A5A5A5A)

    first = cuda_bruteforce._cpu_reference(
        4095,
        cuda_bruteforce.DEFAULT_STRESS_ROUNDS,
        seed0,
        seed1,
    )
    second = cuda_bruteforce._cpu_reference(
        4095,
        cuda_bruteforce.DEFAULT_STRESS_ROUNDS,
        seed0,
        seed1,
    )

    assert first == second
    assert all(0 <= value <= 0xFFFFFFFF for value in first)


def test_cuda_verify_interval_scales_for_short_runs_and_caps_long_runs() -> None:
    assert cuda_bruteforce._verify_interval_s(3.0) == 1.0
    assert cuda_bruteforce._verify_interval_s(5.0) == 5.0 / 3.0
    assert cuda_bruteforce._verify_interval_s(150.0) == 5.0


class _FakeCudaLib:
    """CUDA ordinals in FASTEST_FIRST order on a 3080/5090/3080 host."""

    def __init__(self, ordinal_bus_ids: list[str]) -> None:
        self.ordinal_bus_ids = ordinal_bus_ids
        self.get_calls: list[int] = []

    def cuDeviceGetByPCIBusId(self, device_ref, bus_id: bytes) -> int:
        wanted = cuda_bruteforce._normalize_pci_bus_id(bus_id.decode())
        for ordinal, candidate in enumerate(self.ordinal_bus_ids):
            if cuda_bruteforce._normalize_pci_bus_id(candidate) == wanted:
                device_ref._obj.value = ordinal
                return 0
        return 101  # CUDA_ERROR_INVALID_DEVICE

    def cuDeviceGet(self, device_ref, ordinal: int) -> int:
        self.get_calls.append(int(ordinal))
        device_ref._obj.value = int(ordinal)
        return 0

    def cuDeviceGetPCIBusId(self, buffer, _length: int, device: int) -> int:
        buffer.value = self.ordinal_bus_ids[int(device)].upper().encode()
        return 0


class _FakeCudaDriver:
    def __init__(self, lib: _FakeCudaLib) -> None:
        self.lib = lib

    def check(self, result: int, name: str) -> None:
        if int(result) != 0:
            raise cuda_bruteforce.CudaDriverError(f"{name} failed: {result}")


_RIG_NVML_BUS_IDS = ["0000:08:00.0", "0000:0d:00.0", "0000:0e:00.0"]
# FASTEST_FIRST: the 5090 (bus 0d) is CUDA ordinal 0.
_RIG_CUDA_ORDER = ["0000:0d:00.0", "0000:08:00.0", "0000:0e:00.0"]


def test_normalize_pci_bus_id_accepts_nvml_and_cuda_spellings() -> None:
    assert (
        cuda_bruteforce._normalize_pci_bus_id("00000000:0E:00.0") == "0000:0e:00.0"
    )
    assert cuda_bruteforce._normalize_pci_bus_id("0000:0e:00.0") == "0000:0e:00.0"
    assert cuda_bruteforce._normalize_pci_bus_id("garbage") == ""


@pytest.mark.parametrize("nvml_index", [0, 1, 2])
def test_resolve_cuda_device_follows_nvml_index_not_cuda_ordinal(
    monkeypatch, nvml_index: int
) -> None:
    monkeypatch.setattr(
        cuda_bruteforce,
        "_nvml_pci_bus_id",
        lambda index: _RIG_NVML_BUS_IDS[int(index)],
    )
    lib = _FakeCudaLib(_RIG_CUDA_ORDER)

    device = cuda_bruteforce._resolve_cuda_device(_FakeCudaDriver(lib), nvml_index)

    assert _RIG_CUDA_ORDER[device.value] == _RIG_NVML_BUS_IDS[nvml_index]
    assert lib.get_calls == []


def test_resolve_cuda_device_refuses_to_guess_without_nvml(monkeypatch) -> None:
    monkeypatch.setattr(cuda_bruteforce, "_nvml_pci_bus_id", lambda _index: "")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)

    with pytest.raises(cuda_bruteforce.CudaDriverError, match="refusing to guess"):
        cuda_bruteforce._resolve_cuda_device(
            _FakeCudaDriver(_FakeCudaLib(_RIG_CUDA_ORDER)), 1
        )


def test_resolve_cuda_device_uses_ordinal_when_pci_order_is_forced(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cuda_bruteforce, "_nvml_pci_bus_id", lambda _index: "")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    lib = _FakeCudaLib(_RIG_NVML_BUS_IDS)

    device = cuda_bruteforce._resolve_cuda_device(_FakeCudaDriver(lib), 2)

    assert device.value == 2
    assert lib.get_calls == [2]


def test_resolve_cuda_device_rejects_mismatched_readback(monkeypatch) -> None:
    monkeypatch.setattr(
        cuda_bruteforce, "_nvml_pci_bus_id", lambda _index: "0000:0e:00.0"
    )
    lib = _FakeCudaLib(_RIG_CUDA_ORDER)
    lib.cuDeviceGetPCIBusId = (  # type: ignore[method-assign]
        lambda buffer, _length, _device: setattr(buffer, "value", b"0000:08:00.0")
        or 0
    )

    with pytest.raises(cuda_bruteforce.CudaDriverError, match="expected 0000:0e"):
        cuda_bruteforce._resolve_cuda_device(_FakeCudaDriver(lib), 2)
