"""Basic I/O helpers for dataset discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pandas as pd


CORE_GAMEPLAY_COLUMNS = [
    "loggingTime(txt)",
    "loggingSample(N)",
    "accelerometerTimestamp_sinceReboot(s)",
    "accelerometerAccelerationX(G)",
    "accelerometerAccelerationY(G)",
    "accelerometerAccelerationZ(G)",
    "gyroTimestamp_sinceReboot(s)",
    "gyroRotationX(rad/s)",
    "gyroRotationY(rad/s)",
    "gyroRotationZ(rad/s)",
    "motionTimestamp_sinceReboot(s)",
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
    "motionGravityX(G)",
    "motionGravityY(G)",
    "motionGravityZ(G)",
]

REBOOT_TIME_COLUMNS = [
    "accelerometerTimestamp_sinceReboot(s)",
    "gyroTimestamp_sinceReboot(s)",
    "motionTimestamp_sinceReboot(s)",
]


@dataclass(frozen=True)
class TrimWindow:
    """Inclusive start and exclusive end window in elapsed minutes."""

    start_min: float
    end_min: float


@dataclass(frozen=True)
class TrimSpec:
    """Windows to keep and windows to remove within those keeps."""

    keep_start_min: float
    keep_end_min: float
    remove_windows: tuple[TrimWindow, ...] = ()


def iter_session_dirs(raw_root: str | Path = "data/raw") -> list[Path]:
    """Return session directories under the raw data root."""
    root = Path(raw_root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def load_csv(path: str | Path, device_id: str):
    """Load a CSV and stamp it with a device identifier."""
    frame = pd.read_csv(path)
    frame["device_id"] = device_id
    return frame


def load_game_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a raw game CSV, optionally selecting a subset of columns."""
    usecols = columns if columns is not None else None
    frame = pd.read_csv(path, usecols=usecols)
    frame["loggingTime(txt)"] = pd.to_datetime(frame["loggingTime(txt)"], format="mixed")
    return frame.sort_values("loggingTime(txt)").reset_index(drop=True)


def load_hyperimu_csv(path: str | Path, timezone_name: str = "America/Boise") -> pd.DataFrame:
    """Load a HyperIMU CSV with metadata header rows and synthetic timestamps."""

    csv_path = Path(path)
    header_lines = csv_path.read_text(encoding="utf-8").splitlines()[:4]
    if len(header_lines) < 4:
        raise ValueError(f"{csv_path} does not look like a HyperIMU export.")

    metadata_line = header_lines[1].strip()
    sampling_match = re.search(r"Sampling Rate:(\d+(?:\.\d+)?)ms", metadata_line)
    if sampling_match is None:
        raise ValueError(f"Could not parse sampling rate from {metadata_line!r}.")
    sample_period_s = float(sampling_match.group(1)) / 1000.0

    date_text = metadata_line.split(",")[0].replace("@ Date:", "").strip()
    for tz_abbrev in ("MDT", "MST", "UTC"):
        date_text = date_text.replace(f" {tz_abbrev} ", " ")
    start_naive = datetime.strptime(date_text, "%a %b %d %H:%M:%S %Y")
    start_time = pd.Timestamp(start_naive, tz=ZoneInfo(timezone_name))

    frame = pd.read_csv(csv_path, skiprows=3)
    elapsed_s = pd.Series(range(len(frame)), dtype="float64") * sample_period_s
    frame["loggingTime(txt)"] = start_time + pd.to_timedelta(elapsed_s, unit="s")
    frame["elapsed_s"] = elapsed_s
    return frame


def trim_game_data(frame: pd.DataFrame, spec: TrimSpec) -> pd.DataFrame:
    """Trim a game to the requested elapsed-minute windows and compress removed gaps."""
    trimmed = frame.copy()
    elapsed_min = (
        trimmed["loggingTime(txt)"] - trimmed["loggingTime(txt)"].iloc[0]
    ).dt.total_seconds() / 60.0

    keep_mask = elapsed_min.ge(spec.keep_start_min) & elapsed_min.lt(spec.keep_end_min)
    for window in spec.remove_windows:
        remove_mask = elapsed_min.ge(window.start_min) & elapsed_min.lt(window.end_min)
        keep_mask &= ~remove_mask

    result = trimmed.loc[keep_mask].copy()
    removed_before_row_s = pd.Series(0.0, index=trimmed.index, dtype="float64")
    for window in spec.remove_windows:
        duration_s = (window.end_min - window.start_min) * 60.0
        removed_before_row_s.loc[elapsed_min >= window.end_min] += duration_s

    removed_s_for_kept_rows = removed_before_row_s.loc[result.index]
    result["loggingTime(txt)"] = result["loggingTime(txt)"] - pd.to_timedelta(
        removed_s_for_kept_rows, unit="s"
    )
    for column in REBOOT_TIME_COLUMNS:
        if column in result.columns:
            result[column] = result[column] - removed_s_for_kept_rows.to_numpy()

    result["elapsed_min_from_trim_start"] = (
        result["loggingTime(txt)"] - result["loggingTime(txt)"].iloc[0]
    ).dt.total_seconds() / 60.0
    return result.reset_index(drop=True)
