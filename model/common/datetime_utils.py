"""Filename datetime parsing and DUNEX frame-rate lookup."""
import re
from datetime import datetime

_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


def parse_datetime_from_filename(filename):
    """Extract the required ``YYYYmmddTHHMMSSZ`` stamp from an Argus filename."""
    match = _STAMP_RE.search(filename)
    if match is None:
        raise ValueError(
            f"Argus filename does not contain a YYYYmmddTHHMMSSZ timestamp: {filename}"
        )
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")


def get_time_resolution_for_date(dt):
    """Frame rate for a DUNEX capture, keyed on its date.

    September 19, 2021 videos were recorded at 1 Hz (1.0 s/frame); all other
    videos at 2 Hz (0.5 s/frame).

    Returns ``(dt_seconds, fps_hz)``.
    """
    if dt.date() == datetime(2021, 9, 19).date():
        return 1.0, 1.0  # 1 Hz frame rate
    return 0.5, 2.0  # 2 Hz frame rate (default)
