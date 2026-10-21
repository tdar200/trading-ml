"""Real Chromium workflow. Run: pytest tests/e2e/test_rule_discovery_browser.py."""
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def app(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("discovery-browser")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (directory / "server.log").open("w")
    env = {**os.environ, "TRADING_DISCOVERY_RESULTS": str(directory / "runs")}
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", getattr(request, "param", "dashboard/finder_app.py"),
        "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urlopen(url + "/_stcore/health", timeout=1) as response:
                    if response.status == 200: break
            except OSError:
                time.sleep(.1)
        else:
            raise AssertionError((directory / "server.log").read_text())
        yield url, directory
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()


@pytest.fixture
def page(app, request):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=shutil.which('google-chrome') or None)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.set_default_timeout(15000)
        page.goto(app[0])
        expect(page.locator(".st-key-rd_source")).to_be_attached(timeout=30000)
        if page.get_by_role("tab", name="Rule Strategy Finder", exact=True).count():
            page.get_by_role("tab", name="Rule Strategy Finder", exact=True).click()
        expect(page.get_by_role("heading", name="Rule Strategy Finder", exact=True)).to_be_visible(timeout=30000)
        yield page
        artifacts = ROOT / "results" / "rule_discovery_e2e"
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(artifacts / f"{request.node.name}.png"), full_page=True)
        page.get_by_role("heading", name="Rule Strategy Finder", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(artifacts / f"{request.node.name}_form.png"), full_page=True)
        context.tracing.stop(path=str(artifacts / f"{request.node.name}.zip"))
        browser.close()


def select_source(page, source):
    page.locator(".st-key-rd_source").get_by_role("combobox").click()
    page.get_by_role("option", name=source, exact=True).click()
    settle(page)


def settle(page):
    # Allow the WebSocket widget event to reach Streamlit, then wait for its
    # rerun to complete before interacting with the next generated widget.
    page.wait_for_timeout(150)
    expect(page.get_by_test_id("stApp")).to_have_attribute("data-test-script-state", "notRunning", timeout=30000)


def fill(page, label, value):
    field = page.get_by_label(label, exact=True)
    field.fill(value)
    field.press("Tab")
    settle(page)
    if field.get_attribute("type") == "number":
        assert float(field.input_value()) == float(value)
    else:
        expect(field).to_have_value(value)


def test_search_replay_export_and_reload(page, app):
    select_source(page, "Synthetic demo")
    for key, option in (("rd_instruments", "MSFT"), ("rd_timeframes", "1d")):
        page.locator(f".st-key-{key}").get_by_role("combobox").click()
        page.get_by_role("option", name=option, exact=True).click()
        page.keyboard.press("Escape")
        settle(page)
    fill(page, "Indicator periods (comma-separated)", "7")
    fill(page, "Trend MA periods (0 disables)", "0")
    fill(page, "Holding periods in bars", "4")
    fill(page, "Execution cost per side (bps)", "3")
    page.get_by_role("button", name="Start discovery", exact=True).click()
    expect(page.get_by_text("Search complete", exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_text("Verdict: DEMO_ONLY", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="Exact trade ledger", exact=True)).to_be_visible()
    page.get_by_role("button", name="Replay saved candidate", exact=True).click()
    expect(page.get_by_text("Replay verified: trades, costs and equity exactly match saved evidence", exact=True)).to_be_visible()
    with page.expect_download() as download:
        page.get_by_role("button", name="Download run JSON", exact=True).click()
    saved = download.value.path()
    report = json.loads(Path(saved).read_text())
    assert report["candidate_count"] == 8
    assert {r["instrument"] for r in report["rules"]} == {"AAPL", "MSFT"}
    assert {r["timeframe"] for r in report["rules"]} == {"1h", "1d"}
    assert report["cost_bps"] == 3
    assert report["data_source"] == "Synthetic demo"
    assert len(report["folds"]) == 3
    assert report["deployment_approved"] is False
    evidence = report["evaluation"] or report["preview"]
    assert len(evidence["trades"]) > 0
    with page.expect_download() as download:
        page.get_by_role("button", name="Download trades CSV", exact=True).click()
    assert "net_pnl" in Path(download.value.path()).read_text()
    page.reload()
    expect(page.get_by_text("Verdict: DEMO_ONLY", exact=True)).to_be_visible()
    page.get_by_role("button", name="Replay saved candidate", exact=True).click()
    expect(page.get_by_text("Replay verified: trades, costs and equity exactly match saved evidence", exact=True)).to_be_visible()


def test_invalid_configuration_disables_start(page):
    fill(page, "Indicator periods (comma-separated)", "bad")
    expect(page.get_by_role("button", name="Start discovery", exact=True)).to_be_disabled()
    expect(page.get_by_text("Invalid search:", exact=False)).to_be_visible()


def test_missing_data_does_not_show_previous_winner(page, app):
    fill(page, "Local intraday data directory", str(app[1] / "missing-data"))
    page.get_by_role("button", name="Start discovery", exact=True).click()
    expect(page.get_by_text("No local bars for AAPL 1h.", exact=False)).to_be_visible()
    expect(page.get_by_role("heading", name="Exact trade ledger", exact=True)).to_have_count(0)
    expect(page.get_by_text("Verdict: DEMO_ONLY", exact=True)).to_have_count(0)


def test_cancelled_run_does_not_show_previous_winner(page):
    select_source(page, "Synthetic demo")
    fill(page, "Holding periods in bars", ",".join(str(i) for i in range(1, 61)))
    page.get_by_role("button", name="Start discovery", exact=True).click()
    page.get_by_role("button", name="Cancel search", exact=True).click()
    expect(page.get_by_text("Search cancelled; no result published", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name="Exact trade ledger", exact=True)).to_have_count(0)


def test_rejected_search_keeps_final_period_sealed(page):
    select_source(page, "Synthetic demo")
    fill(page, "Minimum trades per validation window", "1000")
    fill(page, "Indicator periods (comma-separated)", "7")
    fill(page, "Trend MA periods (0 disables)", "0")
    fill(page, "Holding periods in bars", "4")
    page.get_by_role("button", name="Start discovery", exact=True).click()
    expect(page.get_by_text("Search complete", exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role("heading", name="Development replay — final period sealed", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="Final evaluation", exact=True)).to_have_count(0)


@pytest.mark.parametrize("app", ["dashboard/app.py"], indirect=True)
def test_main_dashboard_exposes_finder(page):
    expect(page.get_by_role("tab", name="Rule Strategy Finder", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Start discovery", exact=True)).to_be_visible()
