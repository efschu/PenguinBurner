"""Top-level PenguinBurner CLI entry flow.

This module handles daemon flags and early profile validation before running the selected command.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from common.cli_output import enable_cli_output_wrapping
from profiles.uv.profile_store import read_auto_uv_profiles, resolve_auto_uv_profile
from profiles.uv.profile_tiers import (
    available_adaptive_tiers,
    resolve_profile_tier_profiles,
)
from runtime.daemon_client import apply_runtime_intent, daemon_status
from runtime.runtime_spec import runtime_intent_from_argv
from runtime.support.runtime_debug import debug_exception, log
from runtime.support.runtime_service import (
    DEFAULT_DAEMON_SOCKET,
    install_systemd_service,
    migrate_to_daemon_service,
    parse_runtime_flags,
    reexec_daemon_lifecycle_with_root,
    running_under_systemd_service,
    uninstall_systemd_service,
)

from .runtime_profile_argument import (
    runtime_profile_selector_allows_unverified_from_argv,
    runtime_profile_selector_from_argv,
)


def dispatch_cli(
    *,
    program_file: str | Path,
    main_callback: Callable[..., object],
    argv: list[str] | None = None,
) -> int:
    enable_cli_output_wrapping()
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        runtime_flags = parse_runtime_flags(raw_argv)
        runtime_argv = runtime_flags["passthrough"]
        if runtime_flags["daemon_status"]:
            status = daemon_status(socket_path=DEFAULT_DAEMON_SOCKET)
            print(json.dumps(status, indent=2), flush=True)
            return 0
        if runtime_flags["restore_stock"]:
            # Headless recovery: reset the GPU to stock now and make stock the
            # boot state. Kept GUI-independent so it works when a saved
            # profile misbehaves or the desktop session is unusable.
            from profiles.uv.profile_store import STOCK_PROFILE_SELECTOR

            stock_intent: dict = {"profile_selector": STOCK_PROFILE_SELECTOR}
            # Honour an explicit --gpu-index so multi-GPU hosts can restore
            # one specific card; without it the configured GPU is used.
            restore_gpu_index = runtime_intent_from_argv(runtime_argv).get(
                "gpu_index"
            )
            if restore_gpu_index is not None:
                stock_intent["gpu_index"] = int(restore_gpu_index)
            try:
                _apply_runtime_intent_through_daemon(
                    stock_intent,
                    persist_on_startup=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "restore-stock could not reach the penguin-burnerd "
                    f"service: {exc}\n"
                    "Start the service first:  sudo systemctl start "
                    "penguin-burnerd.service\n"
                    "Or keep any saved profile from applying at boot at all:  "
                    "sudo systemctl disable --now penguin-burnerd.service",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            log(
                "GPU restored to stock; stock is now the boot state. "
                "Saved profiles were kept and can be re-applied from the GUI."
            )
            return 0
        if (
            runtime_flags["install_systemd_service"]
            or runtime_flags["uninstall_systemd_service"]
            or runtime_flags["migrate_to_daemon"]
        ) and os.geteuid() != 0:
            # Daemon-lifecycle commands need root, but a bare sudo on a pip
            # --user install dies on import before product code runs. Elevate
            # this one invocation ourselves (pkexec/sudo) with the invoking
            # user's import root preserved.
            return reexec_daemon_lifecycle_with_root(raw_argv)
        if runtime_flags["migrate_to_daemon"]:
            migrate_to_daemon_service(
                program_file,
                socket_path=DEFAULT_DAEMON_SOCKET,
                log=log,
            )
            return 0
        _require_selected_profile_exists(runtime_argv)
        _require_adaptive_profiles_available(runtime_flags, runtime_argv)
        _reject_auto_uv_scan_in_background(runtime_flags, runtime_argv)
        if runtime_flags["install_systemd_service"]:
            # Always run the real install: it refreshes the canonical /var/opt
            # daemon binary, rewrites the unit, and restarts the service. Probing the
            # socket and "applying instead" when a daemon answered left users
            # running a stale daemon with success-looking output (issue #30).
            install_systemd_service(
                program_file,
                runtime_argv,
                journal_hours=runtime_flags["journal_hours"],
                log=log,
            )
        elif runtime_flags["uninstall_systemd_service"]:
            uninstall_systemd_service(log=log)
        elif runtime_flags["daemonize"]:
            _apply_runtime_through_daemon(runtime_argv, persist_on_startup=False)
        else:
            main_callback(runtime_argv, journal_hours=runtime_flags["journal_hours"])
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        debug_exception("fatal error", exc)
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


def _apply_runtime_through_daemon(
    runtime_argv: list[str],
    *,
    persist_on_startup: bool,
) -> None:
    _apply_runtime_intent_through_daemon(
        runtime_intent_from_argv(runtime_argv),
        persist_on_startup=persist_on_startup,
    )


def _apply_runtime_intent_through_daemon(
    intent: dict,
    *,
    persist_on_startup: bool,
) -> None:
    result = apply_runtime_intent(
        intent,
        persist_on_startup=bool(persist_on_startup),
        socket_path=DEFAULT_DAEMON_SOCKET,
    )
    print(json.dumps(result, indent=2), flush=True)


def _require_selected_profile_exists(runtime_argv: list[str]) -> None:
    from profiles.uv.profile_store import STOCK_PROFILE_SELECTOR

    selector = runtime_profile_selector_from_argv(runtime_argv)
    if not selector or selector == STOCK_PROFILE_SELECTOR:
        # Empty = latest; the stock sentinel = run at stock. Neither names a
        # saved profile to validate.
        return
    if resolve_auto_uv_profile(
        selector,
        allow_unverified=runtime_profile_selector_allows_unverified_from_argv(
            runtime_argv
        ),
    ) is None:
        raise RuntimeError(f"Auto-UV profile not found: {selector}")


def _require_adaptive_profiles_available(
    runtime_flags: dict,
    runtime_argv: list[str],
) -> None:
    if "--adaptive-auto-uv" not in runtime_argv:
        return
    if not (
        runtime_flags["daemonize"] or runtime_flags["install_systemd_service"]
    ):
        return
    profiles = read_auto_uv_profiles()
    tiers = available_adaptive_tiers(resolve_profile_tier_profiles(profiles))
    if tiers:
        return
    raise RuntimeError(
        "Adaptive Auto-UV systemd mode requires at least one saved verified "
        "Auto-UV profile in an available adaptive tier; currently available: "
        "none. Run Auto-UV or assign a profile with "
        "--assign-auto-uv-tier PROFILE TIER."
    )


def _reject_auto_uv_scan_in_background(runtime_flags: dict, runtime_argv: list[str]) -> None:
    auto_uv_requested = "--auto-uv-voltage-scan" in runtime_argv
    if auto_uv_requested and (
        runtime_flags["daemonize"]
        or runtime_flags["install_systemd_service"]
        or running_under_systemd_service()
    ):
        raise RuntimeError(
            "Auto-UV scans are foreground-only; run the scan directly first, "
            "then daemonize normal runtime after the final curve is saved"
        )
