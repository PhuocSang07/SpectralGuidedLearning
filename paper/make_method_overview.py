"""Render figures/src/method_overview.html to figures/method_overview.pdf (Chromium, vector)."""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SRC = HERE / "figures/src/method_overview.html"
OUT = HERE / "figures/method_overview.pdf"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=__import__("os").environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium"))
    page = b.new_page(viewport={"width": 1700, "height": 800})
    page.goto(SRC.as_uri())
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready")
    page.pdf(path=str(OUT), width="1700px", height="800px", print_background=True)
    page.screenshot(path=str(HERE / "figures/src/method_overview_preview.png"))
    b.close()
print("wrote", OUT)
