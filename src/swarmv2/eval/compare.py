"""Side-by-side comparison of two evaluation reports (ported from swarmflow.evaluate.compare_reports)."""
import json
from pathlib import Path

from ..io.files import read_json, write_json_atomic


def _metric(row, key):
    value = row.get(key)
    if isinstance(value, dict):
        return value.get("rate", value.get("mean"))
    return value


def compare_reports(paths, output=None, keys=("success", "slot_rmse_m", "min_clearance_m", "tilt_jitter_deg")):
    paths = [Path(p) for p in paths]
    if len(paths) != 2:
        raise ValueError("compare requires exactly two report paths")
    loaded = []
    for path in paths:
        try:
            loaded.append(read_json(path))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error}") from error
    left, right = (item.get("summary") or {} for item in loaded)
    summary = {}
    for name in dict.fromkeys([*left, *right]):
        a, b = left.get(name, {}), right.get(name, {})
        entry = {}
        for key in keys:
            pair = [_metric(a, key), _metric(b, key)]
            if pair == [None, None]:
                continue
            entry[key] = pair
            if None not in pair:
                entry[f"delta_{key}"] = pair[1] - pair[0]
        summary[name] = entry
    report = {"reports": [p.as_posix() for p in paths], "summary": summary}
    if output:
        write_json_atomic(output, report)
    return report
