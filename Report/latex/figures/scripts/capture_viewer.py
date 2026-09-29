"""Capture screenshots of the SwarmFlow viewer flying the final (V3) model.

Start the viewer first:
    .venv/bin/python -m swarmflow view --checkpoint models/previous-runs/run3-pipeline-v3/showcase/best.pt --port 8766
Then:
    PYTHONPATH=/tmp/pwpkgs .venv/bin/python Report/latex/figures/scripts/capture_viewer.py
"""
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8766/"
OUT = Path(__file__).resolve().parents[1] / "viewer"
OUT.mkdir(parents=True, exist_ok=True)
TIME = "(() => { const m = document.getElementById('simulation-time').textContent.match(/([\\d.]+)\\s*s/); return m ? parseFloat(m[1]) : 0; })()"


def hide_note(page):
    page.evaluate("document.getElementById('learning-note').hidden = true")


def prepare(page, scenario, seed, zoom=1.45):
    page.select_option("#controller", "learned")
    page.wait_for_timeout(600)
    page.select_option("#scenario", scenario)
    page.wait_for_timeout(600)
    page.fill("#seed", str(seed))
    # Benchmark spacing (Config.spacing = 1.25 m); the viewer's presets are 0.85/1.1/1.45/1.9 m.
    page.evaluate("""() => { const s = document.getElementById('compaction');
        if (![...s.options].some(o => o.value === '1.25')) s.add(new Option('Benchmark (1.25 m)', '1.25'));
        s.value = '1.25'; }""")
    page.click("#reset")
    page.wait_for_timeout(1200)
    page.click("#orbit")
    page.evaluate(f"zoom = {zoom}")
    hide_note(page)


def run_until(page, when):
    if not page.evaluate("playing"):
        page.click("#play")
    page.wait_for_function(f"() => {TIME} >= {when} || !playing", timeout=120000, polling=50)
    if page.evaluate("playing"):
        page.click("#play")
    page.wait_for_timeout(500)
    hide_note(page)
    return page.evaluate(TIME)


def run_to_end(page):
    if not page.evaluate("playing"):
        page.click("#play")
    page.wait_for_function("() => !playing", timeout=120000, polling=100)
    page.wait_for_timeout(700)
    hide_note(page)


def panel_shot(page, path):
    """Viewport + metric strip + slot-error timeline, without the empty area below."""
    top = page.locator(".flight-toolbar").bounding_box()
    bottom = page.locator(".timeline").bounding_box()
    page.screenshot(path=path, full_page=True,
                    clip={"x": top["x"], "y": top["y"], "width": top["width"],
                          "height": bottom["y"] + bottom["height"] - top["y"]})


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        page.goto(URL)
        page.wait_for_selector("#play:not([disabled])", timeout=30000)
        page.wait_for_timeout(1500)

        prepare(page, "crossing", 100000)
        shots = {}
        for label, when in [("a_formation", 2.0), ("b_approach", 7.0), ("c_closest", 8.3), ("d_recovered", 14.0)]:
            shots[label] = run_until(page, when)
            page.locator(".viewport").screenshot(path=str(OUT / f"crossing_{label}.png"))
            if label == "c_closest":
                page.click("#axis")
                page.evaluate("zoom = 1.25")
                page.wait_for_timeout(500)
                page.locator(".viewport").screenshot(path=str(OUT / "crossing_axis_views.png"))
                page.click("#orbit")
                page.evaluate("zoom = 1.45")
                page.wait_for_timeout(300)
                panel_shot(page, str(OUT / "crossing_flight_panel.png"))
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(300)
                box = page.locator(".workspace").bounding_box()
                page.screenshot(path=str(OUT / "viewer_overview.png"), full_page=True,
                                clip={"x": 0, "y": 0, "width": 1500, "height": box["y"] + box["height"] + 12})
        run_to_end(page)
        panel_shot(page, str(OUT / "crossing_end_panel.png"))

        prepare(page, "head_on", 100000)
        for label, when in [("closest", 7.6), ("recovered", 13.0)]:
            shots["head_on_" + label] = run_until(page, when)
            page.locator(".viewport").screenshot(path=str(OUT / f"head_on_{label}.png"))

        prepare(page, "double_crossing", 100000)
        shots["double_second"] = run_until(page, 16.5)
        page.locator(".viewport").screenshot(path=str(OUT / "double_crossing_second.png"))
        run_to_end(page)
        panel_shot(page, str(OUT / "double_crossing_end_panel.png"))

        page.locator("#model").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        page.locator(".model-charts").screenshot(path=str(OUT / "model_training_charts.png"))
        browser.close()
        print(shots)


if __name__ == "__main__":
    main()
