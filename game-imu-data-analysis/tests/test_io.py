from pathlib import Path

import pandas as pd

from imu_pipeline.io import TrimSpec, TrimWindow, iter_session_dirs, trim_game_data


def test_iter_session_dirs_returns_sorted_directories(tmp_path: Path) -> None:
    (tmp_path / "b_session").mkdir()
    (tmp_path / "a_session").mkdir()
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    result = iter_session_dirs(tmp_path)

    assert [path.name for path in result] == ["a_session", "b_session"]


def test_trim_game_data_keeps_requested_window() -> None:
    frame = pd.DataFrame(
        {
            "loggingTime(txt)": pd.to_datetime(
                [
                    "2026-03-24T17:00:00",
                    "2026-03-24T17:01:00",
                    "2026-03-24T17:02:00",
                    "2026-03-24T17:03:00",
                    "2026-03-24T17:04:00",
                ]
            ),
            "value": [0, 1, 2, 3, 4],
        }
    )

    result = trim_game_data(frame, TrimSpec(keep_start_min=1.0, keep_end_min=4.0))

    assert result["value"].tolist() == [1, 2, 3]
    assert result["elapsed_min_from_trim_start"].tolist() == [0.0, 1.0, 2.0]


def test_trim_game_data_removes_break_window() -> None:
    frame = pd.DataFrame(
        {
            "loggingTime(txt)": pd.to_datetime(
                [
                    "2026-03-24T17:00:00",
                    "2026-03-24T17:01:00",
                    "2026-03-24T17:02:00",
                    "2026-03-24T17:03:00",
                    "2026-03-24T17:04:00",
                    "2026-03-24T17:05:00",
                ]
            ),
            "value": [0, 1, 2, 3, 4, 5],
        }
    )

    result = trim_game_data(
        frame,
        TrimSpec(
            keep_start_min=0.0,
            keep_end_min=6.0,
            remove_windows=(TrimWindow(2.0, 4.0),),
        ),
    )

    assert result["value"].tolist() == [0, 1, 4, 5]
    assert result["loggingTime(txt)"].tolist() == list(
        pd.to_datetime(
            [
                "2026-03-24T17:00:00",
                "2026-03-24T17:01:00",
                "2026-03-24T17:02:00",
                "2026-03-24T17:03:00",
            ]
        )
    )
