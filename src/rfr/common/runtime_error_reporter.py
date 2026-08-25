from __future__ import annotations

import shlex
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from rfr.search.runtime.elastic_gpu import (
    ELASTIC_GPU_RESTART_EXIT_CODE,
    ElasticGPUFailure,
    ElasticGPURestartRequested,
    write_elastic_gpu_failure_record,
)


ERROR_SUMMARY_FILENAME = "error_summary.txt"


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def extract_cli_option(
    argv: Sequence[str],
    option_names: Iterable[str],
) -> str:
    normalized = set()
    for name in option_names:
        name = str(name).strip()
        if not name:
            continue
        normalized.add(name if name.startswith("--") else f"--{name}")

    items = list(argv)
    for idx, token in enumerate(items):
        if token in normalized:
            if idx + 1 < len(items):
                return str(items[idx + 1]).strip()
            return ""
        for name in normalized:
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix):].strip()
    return ""


def resolve_logs_dir(run_output_dir: Optional[str]) -> Optional[Path]:
    text = str(run_output_dir or "").strip()
    if not text:
        return None
    return Path(text).expanduser() / "logs"


def resolve_error_summary_path(
    run_output_dir: Optional[str],
    filename: str = ERROR_SUMMARY_FILENAME,
) -> Optional[Path]:
    logs_dir = resolve_logs_dir(run_output_dir)
    if logs_dir is None:
        return None
    return logs_dir / filename


def clear_error_summary(
    run_output_dir: Optional[str],
    filename: str = ERROR_SUMMARY_FILENAME,
) -> Optional[Path]:
    path = resolve_error_summary_path(run_output_dir, filename=filename)
    if path is not None and path.exists():
        path.unlink()
    return path


def format_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in argv)


def read_text_tail(path: Union[Path, str], max_lines: int = 80) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""

    tail_lines = deque(maxlen=max(1, int(max_lines)))
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tail_lines.append(line.rstrip("\n"))
    return "\n".join(tail_lines)


def write_error_summary(
    run_output_dir: Optional[str],
    *,
    program_name: str,
    status: str,
    message: str,
    argv: Optional[Sequence[str]] = None,
    traceback_text: str = "",
    exit_code: Optional[int] = None,
    signal_name: str = "",
    extra_sections: Optional[Iterable[tuple[str, str]]] = None,
    summary_path: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    path = Path(summary_path) if summary_path else resolve_error_summary_path(run_output_dir)
    if path is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Time", now_ts()),
        ("Program", program_name),
        ("Status", status),
        ("Message", message),
    ]
    if run_output_dir:
        sections.append(("Run Output Dir", str(run_output_dir)))
    if exit_code is not None:
        sections.append(("Exit Code", str(exit_code)))
    if signal_name:
        sections.append(("Signal", signal_name))
    if argv:
        sections.append(("Command", format_command(argv)))
    if traceback_text:
        sections.append(("Traceback", traceback_text.rstrip()))
    if extra_sections:
        for title, body in extra_sections:
            if body:
                sections.append((str(title), str(body).rstrip()))

    chunks = [f"[{title}]\n{body}\n" for title, body in sections if body is not None]
    text = "\n".join(chunks).rstrip() + "\n"
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(text)
    return path


def system_exit_code(raw_code) -> int:
    if raw_code is None:
        return 0
    if isinstance(raw_code, bool):
        return int(raw_code)
    if isinstance(raw_code, int):
        return raw_code
    return 1


def run_fire_entrypoint(
    fire_module,
    target,
    *,
    program_name: str,
    option_names: Sequence[str] = ("output_dir", "output-dir"),
) -> None:
    argv = list(sys.argv)
    run_output_dir = extract_cli_option(argv[1:], option_names)
    clear_error_summary(run_output_dir)

    try:
        fire_module.Fire(target)
    except (ElasticGPUFailure, ElasticGPURestartRequested) as exc:
        write_elastic_gpu_failure_record(run_output_dir, exc)
        write_error_summary(
            run_output_dir,
            program_name=program_name,
            status="elastic_gpu_restart",
            message=f"{type(exc).__name__}: {exc}",
            argv=argv,
            exit_code=ELASTIC_GPU_RESTART_EXIT_CODE,
        )
        raise SystemExit(ELASTIC_GPU_RESTART_EXIT_CODE) from None
    except KeyboardInterrupt:
        write_error_summary(
            run_output_dir,
            program_name=program_name,
            status="interrupted",
            message="Received KeyboardInterrupt/SIGINT. The run stopped before completion.",
            argv=argv,
            signal_name="SIGINT",
        )
        raise SystemExit(130) from None
    except SystemExit as exc:
        exit_code = system_exit_code(exc.code)
        if exit_code != 0:
            write_error_summary(
                run_output_dir,
                program_name=program_name,
                status="failed",
                message=f"Process exited with a non-zero status ({exit_code}).",
                argv=argv,
                exit_code=exit_code,
            )
        raise
    except BaseException as exc:  # noqa: BLE001 - preserve full failure context
        write_error_summary(
            run_output_dir,
            program_name=program_name,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            argv=argv,
            traceback_text=traceback.format_exc(),
        )
        raise


def summarize_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def callable_traceback() -> str:
    return traceback.format_exc()
