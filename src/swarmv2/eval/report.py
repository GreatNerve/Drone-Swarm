"""Reproducible JSON + Markdown reports with 95% intervals."""
from ..io.files import write_json_atomic, write_text_atomic
from .metrics import aggregate, learning_checks, selection_score


def build_report(episodes, name, gate=None, metadata=None):
    gate = gate or {}
    summary = aggregate(episodes, gate.get("recovery_limit_s", 3.5))
    return {"name": name, "metadata": metadata or {}, "summary": summary,
            "score": selection_score(summary) if summary else None,
            "checks": learning_checks(episodes, **gate), "episodes": episodes}


def _interval(stat, scale=1.0, digits=3):
    if not stat or stat.get("mean") is None:
        return "-"
    text = f"{stat['mean'] * scale:.{digits}f}"
    if stat.get("low") is not None:
        text += f" [{stat['low'] * scale:.{digits}f}, {stat['high'] * scale:.{digits}f}]"
    return text


def _rate(stat):
    if not stat or stat.get("rate") is None:
        return "-"
    return f"{stat['rate']:.1%} [{stat['low']:.1%}, {stat['high']:.1%}]"


def markdown(report):
    lines = [f"# {report['name']}", ""]
    for key, value in (report.get("metadata") or {}).items():
        lines.append(f"- {key}: {value}")
    if report.get("score") is not None:
        lines.append(f"- selection score: {report['score']:.3f}")
    lines += ["", "| Scenario | Episodes | Collision-free (95% CI) | Slot RMSE m (95% CI) | Min clearance m | "
              "Recovered | Tilt jitter deg | Wind error m/s | Reasoning transitions |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in report["summary"].items():
        clearance = "-" if row["min_clearance_m"] is None else f"{row['min_clearance_m']:.3f}"
        recovered = f"{row['recovered_within_limit']}/{row['events']}" if row["events"] else "-"
        lines.append(f"| {name} | {row['episodes']} | {_rate(row['success'])} | {_interval(row['slot_rmse_m'])} | "
                     f"{clearance} | {recovered} | {_interval(row.get('tilt_jitter_deg'), digits=2)} | "
                     f"{_interval(row.get('wind_tracking_error_mps'), digits=2)} | "
                     f"{_interval(row.get('reasoning_transitions'), digits=1)} |")
    lines += ["", "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in report["checks"]]
    return "\n".join(lines) + "\n"


def write_report(report, json_path, markdown_path=None):
    write_json_atomic(json_path, report)
    if markdown_path:
        write_text_atomic(markdown_path, markdown(report))
    return report
