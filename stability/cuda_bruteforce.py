from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import time

PTX_SOURCE = rb"""
.version 6.0
.target sm_52
.address_size 64

.visible .entry brute_force_kernel(
    .param .u64 out_x,
    .param .u64 out_y,
    .param .u32 n,
    .param .u32 rounds,
    .param .u32 seed0,
    .param .u32 seed1
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [out_x];
    ld.param.u64 %rd2, [out_y];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [rounds];
    ld.param.u32 %r3, [seed0];
    ld.param.u32 %r4, [seed1];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mov.u32 %r8, %nctaid.x;
    mad.lo.u32 %r9, %r6, %r7, %r5;
    mul.lo.u32 %r10, %r7, %r8;

LOOP_I:
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra DONE;

    xor.b32 %r11, %r3, %r9;
    mad.lo.u32 %r12, %r9, 17, %r4;
    mov.u32 %r13, 0;

LOOP_R:
    setp.ge.u32 %p2, %r13, %r2;
    @%p2 bra STORE;

    mad.lo.u32 %r11, %r11, 1664525, 1013904223;
    add.u32 %r11, %r11, %r12;
    shr.u32 %r14, %r11, 13;
    xor.b32 %r12, %r12, %r14;
    mad.lo.u32 %r12, %r12, 22695477, 1;
    shl.b32 %r15, %r12, 7;
    xor.b32 %r11, %r11, %r15;
    shr.u32 %r16, %r11, 17;
    xor.b32 %r12, %r12, %r16;

    add.u32 %r13, %r13, 1;
    bra LOOP_R;

STORE:
    mul.wide.u32 %rd3, %r9, 4;
    add.s64 %rd4, %rd1, %rd3;
    st.global.u32 [%rd4], %r11;
    add.s64 %rd5, %rd2, %rd3;
    st.global.u32 [%rd5], %r12;
    add.u32 %r9, %r9, %r10;
    bra LOOP_I;

DONE:
    ret;
}
"""

DEFAULT_STRESS_ELEMENTS = 4 * 1024 * 1024
DEFAULT_VERIFY_ELEMENTS = 4096
DEFAULT_STRESS_ROUNDS = 8192
DEFAULT_VERIFY_ROUNDS = 256
DEFAULT_STRESS_BATCH_LAUNCHES = 8
MAX_VERIFY_INTERVAL_S = 5.0


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _cpu_reference(index: int, rounds: int, seed0: int, seed1: int) -> tuple[int, int]:
    x = _u32(seed0 ^ index)
    y = _u32(index * 17 + seed1)
    for _ in range(int(rounds)):
        x = _u32(x * 1664525 + 1013904223)
        x = _u32(x + y)
        y = _u32(y ^ (x >> 13))
        y = _u32(y * 22695477 + 1)
        x = _u32(x ^ ((y << 7) & 0xFFFFFFFF))
        y = _u32(y ^ (x >> 17))
    return x, y


def _stress_seed(base_seed: int, launch_index: int, stream_seed: int) -> int:
    return _u32(int(base_seed) + int(launch_index) * 0x9E3779B9 + int(stream_seed))


def _verification_sample_indices(element_count: int) -> list[int]:
    count = int(element_count)
    candidates = [
        0,
        1,
        2,
        7,
        31,
        255,
        1023,
        4095,
        count // 4,
        count // 2,
        (count * 3) // 4,
        count - 1,
    ]
    seen = set()
    indices = []
    for index in candidates:
        if index < 0 or index >= count or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def _verify_interval_s(duration_seconds: float) -> float:
    duration_s = max(1.0, float(duration_seconds))
    return max(1.0, min(MAX_VERIFY_INTERVAL_S, duration_s / 3.0))


class CudaDriverError(RuntimeError):
    pass


class _NvmlPciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


def _normalize_pci_bus_id(bus_id: str) -> str:
    """Return ``dddd:bb:dd.f`` (lowercase, 16-bit domain) or "" if malformed."""
    cleaned = str(bus_id or "").strip().lower()
    try:
        domain_text, bus_text, rest = cleaned.split(":", 2)
        device_text, function_text = rest.split(".", 1)
        domain = int(domain_text, 16) & 0xFFFF
        bus = int(bus_text, 16)
        device = int(device_text, 16)
        function = int(function_text, 16)
    except ValueError:
        return ""
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def _nvml_pci_bus_id(gpu_index: int) -> str:
    """PCI bus id of NVML device ``gpu_index``, or "" when NVML is unusable.

    ``--gpu-index`` is an NVML index everywhere else in PenguinBurner (the
    daemon opens ``nvml.device_by_index``), and NVML enumerates in PCI bus
    order. CUDA's default ordinal order is FASTEST_FIRST, so on hosts with
    mixed GPU models ``cuDeviceGet(gpu_index)`` loads a different card than
    the one being tuned. Resolving through the PCI bus id keeps both aligned.
    """
    try:
        nvml = ctypes.CDLL(
            ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        )
    except OSError:
        return ""
    nvml.nvmlInit_v2.restype = ctypes.c_int
    nvml.nvmlShutdown.restype = ctypes.c_int
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    nvml.nvmlDeviceGetPciInfo_v3.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NvmlPciInfo),
    ]
    nvml.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int
    if nvml.nvmlInit_v2() != 0:
        return ""
    try:
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(int(gpu_index)), ctypes.byref(handle)
        ) != 0:
            return ""
        info = _NvmlPciInfo()
        if nvml.nvmlDeviceGetPciInfo_v3(handle, ctypes.byref(info)) != 0:
            return ""
        return f"{info.domain & 0xFFFF:04x}:{info.bus:02x}:{info.device:02x}.0"
    finally:
        nvml.nvmlShutdown()


def _cuda_device_pci_bus_id(driver: "_CudaDriver", device: int) -> str:
    buffer = ctypes.create_string_buffer(32)
    driver.check(
        driver.lib.cuDeviceGetPCIBusId(buffer, len(buffer), int(device)),
        "cuDeviceGetPCIBusId",
    )
    return _normalize_pci_bus_id(buffer.value.decode("ascii", "replace"))


def _resolve_cuda_device(driver: "_CudaDriver", gpu_index: int) -> ctypes.c_int:
    """Open the CUDA device that is NVML device ``gpu_index``.

    Fails loudly instead of loading another card: a companion load on the
    wrong GPU makes every stability verdict meaningless.
    """
    device = ctypes.c_int()
    target_bus_id = _nvml_pci_bus_id(int(gpu_index))
    if target_bus_id:
        driver.check(
            driver.lib.cuDeviceGetByPCIBusId(
                ctypes.byref(device), target_bus_id.encode("ascii")
            ),
            f"cuDeviceGetByPCIBusId({target_bus_id})",
        )
    else:
        if os.environ.get("CUDA_DEVICE_ORDER", "") != "PCI_BUS_ID":
            raise CudaDriverError(
                "cannot resolve NVML GPU index "
                f"{int(gpu_index)} to a PCI bus id and CUDA_DEVICE_ORDER is not "
                "PCI_BUS_ID; refusing to guess the CUDA device"
            )
        driver.check(
            driver.lib.cuDeviceGet(ctypes.byref(device), int(gpu_index)),
            "cuDeviceGet",
        )
    actual_bus_id = _cuda_device_pci_bus_id(driver, int(device.value))
    if target_bus_id and actual_bus_id != target_bus_id:
        raise CudaDriverError(
            f"CUDA device {int(device.value)} is at PCI {actual_bus_id}, "
            f"expected {target_bus_id} for NVML GPU index {int(gpu_index)}"
        )
    _log(
        f"bound nvml-gpu-index={int(gpu_index)} cuda-ordinal={int(device.value)} "
        f"pci-bus-id={actual_bus_id}"
    )
    return device


class _CudaDriver:
    def __init__(self) -> None:
        library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.lib = ctypes.CDLL(library_name)
        self._bind()

    def _bind(self) -> None:
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        self.lib.cuDeviceGetByPCIBusId.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
        ]
        self.lib.cuDeviceGetByPCIBusId.restype = ctypes.c_int
        self.lib.cuDeviceGetPCIBusId.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDeviceGetPCIBusId.restype = ctypes.c_int
        self.lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.cuCtxCreate_v2.restype = ctypes.c_int
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxDestroy_v2.restype = ctypes.c_int
        self.lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuModuleLoadData.restype = ctypes.c_int
        self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleUnload.restype = ctypes.c_int
        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.cuModuleGetFunction.restype = ctypes.c_int
        self.lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.lib.cuMemAlloc_v2.restype = ctypes.c_int
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemFree_v2.restype = ctypes.c_int
        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuLaunchKernel.restype = ctypes.c_int
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuCtxSynchronize.restype = ctypes.c_int
        cu_get_error_string = getattr(self.lib, "cuGetErrorString", None)
        if cu_get_error_string is not None:
            cu_get_error_string.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            cu_get_error_string.restype = ctypes.c_int

    def check(self, rc: int, context: str) -> None:
        if int(rc) == 0:
            return
        detail = ""
        cu_get_error_string = getattr(self.lib, "cuGetErrorString", None)
        if cu_get_error_string is not None:
            message = ctypes.c_char_p()
            if (
                int(cu_get_error_string(int(rc), ctypes.byref(message))) == 0
                and message.value
            ):
                detail = f": {message.value.decode(errors='replace')}"
        raise CudaDriverError(f"{context} failed rc={int(rc)}{detail}")


def _log(message: str) -> None:
    print(f"cuda-bruteforce: {message}", flush=True)


def _verify_outputs(
    driver: _CudaDriver,
    dev_x: ctypes.c_uint64,
    dev_y: ctypes.c_uint64,
    *,
    element_count: int,
    rounds: int,
    seed0: int,
    seed1: int,
    label: str,
) -> None:
    if int(element_count) <= 0:
        raise CudaDriverError(f"{label} verification needs positive element count")
    if dev_x.value is None or dev_y.value is None:
        raise CudaDriverError(f"{label} verification received null device pointer")
    host_x = ctypes.c_uint32()
    host_y = ctypes.c_uint32()
    sample_indices = _verification_sample_indices(int(element_count))
    checked = []
    for index in sample_indices:
        offset = int(index) * ctypes.sizeof(ctypes.c_uint32)
        driver.check(
            driver.lib.cuMemcpyDtoH_v2(
                ctypes.byref(host_x),
                ctypes.c_uint64(int(dev_x.value) + offset),
                ctypes.sizeof(host_x),
            ),
            f"cuMemcpyDtoH_v2({label}.x[{index}])",
        )
        driver.check(
            driver.lib.cuMemcpyDtoH_v2(
                ctypes.byref(host_y),
                ctypes.c_uint64(int(dev_y.value) + offset),
                ctypes.sizeof(host_y),
            ),
            f"cuMemcpyDtoH_v2({label}.y[{index}])",
        )
        expected_x, expected_y = _cpu_reference(
            index, int(rounds), int(seed0), int(seed1)
        )
        actual_x = int(host_x.value)
        actual_y = int(host_y.value)
        if actual_x != expected_x or actual_y != expected_y:
            raise CudaDriverError(
                f"{label} verification mismatch idx={index} "
                f"x={actual_x} expected_x={expected_x} "
                f"y={actual_y} expected_y={expected_y}"
            )
        checked.append(index)
    _log(f"verified {label} indices={checked}")


def _launch_kernel(
    driver: _CudaDriver,
    function: ctypes.c_void_p,
    *,
    out_x: ctypes.c_uint64,
    out_y: ctypes.c_uint64,
    element_count: int,
    rounds: int,
    seed0: int,
    seed1: int,
    grid_dim: int,
    block_dim: int,
) -> None:
    out_x_param = ctypes.c_uint64(int(out_x.value))
    out_y_param = ctypes.c_uint64(int(out_y.value))
    n_param = ctypes.c_uint32(int(element_count))
    rounds_param = ctypes.c_uint32(int(rounds))
    seed0_param = ctypes.c_uint32(int(seed0))
    seed1_param = ctypes.c_uint32(int(seed1))
    params = (ctypes.c_void_p * 6)(
        ctypes.cast(ctypes.byref(out_x_param), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(out_y_param), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_param), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(rounds_param), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(seed0_param), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(seed1_param), ctypes.c_void_p),
    )
    driver.check(
        driver.lib.cuLaunchKernel(
            function,
            int(grid_dim),
            1,
            1,
            int(block_dim),
            1,
            1,
            0,
            None,
            params,
            None,
        ),
        "cuLaunchKernel",
    )


def run_cuda_bruteforce_test(*, gpu_index: int, duration_seconds: float) -> None:
    driver = _CudaDriver()
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    stress_x = ctypes.c_uint64()
    stress_y = ctypes.c_uint64()
    verify_x = ctypes.c_uint64()
    verify_y = ctypes.c_uint64()
    stress_elements = DEFAULT_STRESS_ELEMENTS
    verify_elements = DEFAULT_VERIFY_ELEMENTS
    stress_rounds = DEFAULT_STRESS_ROUNDS
    verify_rounds = DEFAULT_VERIFY_ROUNDS
    seed0 = 0x13579BDF
    seed1 = 0x2468ACE1
    grid_dim = 256
    block_dim = 256
    launches = 0
    verification_passes = 0
    ptx_buffer = ctypes.create_string_buffer(PTX_SOURCE)
    last_stress_seed0 = seed0
    last_stress_seed1 = seed1

    driver.check(driver.lib.cuInit(0), "cuInit")
    device = _resolve_cuda_device(driver, int(gpu_index))
    driver.check(
        driver.lib.cuCtxCreate_v2(ctypes.byref(context), 0, int(device.value)),
        "cuCtxCreate_v2",
    )
    try:
        driver.check(
            driver.lib.cuModuleLoadData(
                ctypes.byref(module), ctypes.cast(ptx_buffer, ctypes.c_void_p)
            ),
            "cuModuleLoadData",
        )
        driver.check(
            driver.lib.cuModuleGetFunction(
                ctypes.byref(function), module, b"brute_force_kernel"
            ),
            "cuModuleGetFunction",
        )
        driver.check(
            driver.lib.cuMemAlloc_v2(ctypes.byref(stress_x), stress_elements * 4),
            "cuMemAlloc_v2(stress_x)",
        )
        driver.check(
            driver.lib.cuMemAlloc_v2(ctypes.byref(stress_y), stress_elements * 4),
            "cuMemAlloc_v2(stress_y)",
        )
        driver.check(
            driver.lib.cuMemAlloc_v2(ctypes.byref(verify_x), verify_elements * 4),
            "cuMemAlloc_v2(verify_x)",
        )
        driver.check(
            driver.lib.cuMemAlloc_v2(ctypes.byref(verify_y), verify_elements * 4),
            "cuMemAlloc_v2(verify_y)",
        )

        _log(
            f"starting gpu-index={int(gpu_index)} duration={float(duration_seconds):.1f}s "
            f"stress-elements={stress_elements} stress-rounds={stress_rounds}"
        )

        _launch_kernel(
            driver,
            function,
            out_x=verify_x,
            out_y=verify_y,
            element_count=verify_elements,
            rounds=verify_rounds,
            seed0=seed0,
            seed1=seed1,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        driver.check(driver.lib.cuCtxSynchronize(), "cuCtxSynchronize(initial verify)")
        _verify_outputs(
            driver,
            verify_x,
            verify_y,
            element_count=verify_elements,
            rounds=verify_rounds,
            seed0=seed0,
            seed1=seed1,
            label="sanity",
        )
        verification_passes += 1

        start_monotonic = time.monotonic()
        verify_interval_s = _verify_interval_s(float(duration_seconds))
        next_verify_monotonic = start_monotonic + verify_interval_s
        deadline = start_monotonic + max(1.0, float(duration_seconds))
        while time.monotonic() < deadline:
            for _ in range(DEFAULT_STRESS_BATCH_LAUNCHES):
                launch_seed0 = _stress_seed(seed0, launches, 0xA5A5A5A5)
                launch_seed1 = _stress_seed(seed1, launches, 0x5A5A5A5A)
                _launch_kernel(
                    driver,
                    function,
                    out_x=stress_x,
                    out_y=stress_y,
                    element_count=stress_elements,
                    rounds=stress_rounds,
                    seed0=launch_seed0,
                    seed1=launch_seed1,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                )
                last_stress_seed0 = launch_seed0
                last_stress_seed1 = launch_seed1
                launches += 1
            driver.check(driver.lib.cuCtxSynchronize(), "cuCtxSynchronize(stress)")
            now_monotonic = time.monotonic()
            if now_monotonic >= next_verify_monotonic:
                _verify_outputs(
                    driver,
                    stress_x,
                    stress_y,
                    element_count=stress_elements,
                    rounds=stress_rounds,
                    seed0=last_stress_seed0,
                    seed1=last_stress_seed1,
                    label="stress",
                )
                verification_passes += 1
                _launch_kernel(
                    driver,
                    function,
                    out_x=verify_x,
                    out_y=verify_y,
                    element_count=verify_elements,
                    rounds=verify_rounds,
                    seed0=seed0,
                    seed1=seed1,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                )
                driver.check(
                    driver.lib.cuCtxSynchronize(), "cuCtxSynchronize(periodic verify)"
                )
                _verify_outputs(
                    driver,
                    verify_x,
                    verify_y,
                    element_count=verify_elements,
                    rounds=verify_rounds,
                    seed0=seed0,
                    seed1=seed1,
                    label="sanity",
                )
                verification_passes += 1
                next_verify_monotonic = now_monotonic + verify_interval_s

        if launches > 0:
            _verify_outputs(
                driver,
                stress_x,
                stress_y,
                element_count=stress_elements,
                rounds=stress_rounds,
                seed0=last_stress_seed0,
                seed1=last_stress_seed1,
                label="stress",
            )
            verification_passes += 1
        _launch_kernel(
            driver,
            function,
            out_x=verify_x,
            out_y=verify_y,
            element_count=verify_elements,
            rounds=verify_rounds,
            seed0=seed0,
            seed1=seed1,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        driver.check(driver.lib.cuCtxSynchronize(), "cuCtxSynchronize(final verify)")
        _verify_outputs(
            driver,
            verify_x,
            verify_y,
            element_count=verify_elements,
            rounds=verify_rounds,
            seed0=seed0,
            seed1=seed1,
            label="sanity",
        )
        verification_passes += 1
        _log(
            f"completed launches={launches} verifications={verification_passes} "
            f"elapsed={time.monotonic() - start_monotonic:.1f}s"
        )
    finally:
        for pointer in (stress_x, stress_y, verify_x, verify_y):
            pointer_value = pointer.value
            if pointer_value is not None and int(pointer_value) != 0:
                try:
                    driver.lib.cuMemFree_v2(pointer)
                except Exception:  # noqa: BLE001, S110
                    pass
        module_value = module.value
        if module_value is not None and int(module_value) != 0:
            try:
                driver.lib.cuModuleUnload(module)
            except Exception:  # noqa: BLE001, S110
                pass
        context_value = context.value
        if context_value is not None and int(context_value) != 0:
            try:
                driver.lib.cuCtxDestroy_v2(context)
            except Exception:  # noqa: BLE001, S110
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic CUDA brute-force stability workload.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=90.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_cuda_bruteforce_test(
            gpu_index=int(args.gpu_index),
            duration_seconds=float(args.duration_seconds),
        )
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        _log(f"FAILED {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
