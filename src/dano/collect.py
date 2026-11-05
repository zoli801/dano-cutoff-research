"""Download the official archive with a checksum and provenance for every asset.

Raw result workbooks stay local: only anonymous aggregates are published.
HTML and linked video/folder references are retained in the manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://dano.hse.ru"
RAW = ROOT / "data/raw"


def tidy(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def fetch(url: str, destination: Path, max_bytes: int = 200_000_000):
    if destination.exists() and destination.stat().st_size:
        return destination.read_bytes()
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(20, 90), stream=True,
                                    headers={"User-Agent": "DANO-academic-archive/0.1 (local reproducibility research)"})
            response.raise_for_status()
            if int(response.headers.get("Content-Length", 0)) > max_bytes:
                raise ValueError("supplementary asset exceeds 200 MB download limit")
            content = bytearray()
            for chunk in response.iter_content(1024 * 256):
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise ValueError("supplementary asset exceeds 200 MB download limit")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return bytes(content)
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def stage_id(text):
    text = tidy(text)
    if text.startswith("Первый (отборочный)"):
        return "stage1"
    if text.startswith("Второй (отборочный)"):
        return "stage2"
    if "Проектный" in text:
        return "final_project"
    if "Задачный" in text:
        return "final_individual"
    if text.startswith(("Третий (заключительный)", "Заключительный этап")):
        return "final"
    return None


def archive_assets(html: str, source_url: str, default_year=None):
    soup = BeautifulSoup(html, "html.parser")
    year, stage = default_year, "demo"
    rows = []
    for tag in soup.find_all(["h2", "h3", "a"]):
        label = tidy(tag.get_text(" ", strip=True))
        if tag.name in ("h2", "h3"):
            match = re.search(r"(20\d{2})\s*[-–/]\s*(20\d{2})", label)
            if match:
                year = int(match[1])
            parsed = stage_id(label)
            if parsed:
                stage = parsed
            continue
        if year is None or not tag.get("href"):
            continue
        raw_url = urljoin(source_url, tag["href"])
        # Three official anchors accidentally include a literal HTML target attribute.
        url = re.sub(r"\s+target=.*$", "", raw_url)
        if ("/mirror/pubs/share/" not in url and
                label not in ("Базы данных", "Разбор") and
                not re.search(r"\.(pdf|xlsx?|csv|zip)(?:\?|$)", url, re.I)):
            continue
        if not label or tag.find("img"):
            continue
        if re.search(r"\.(xlsx?|csv|zip)$", url, re.I):
            kind = "supplement"
        elif "Разбор" in label:
            kind = "solution"
        elif "Критерии" in label or "критерии" in label:
            kind = "rubric"
        elif "Базы данных" in label:
            kind = "data_page"
        elif re.search(r"вариант|[Зз]адани|[Зз]адач|[Сс]качать", label):
            kind = "problem"
        else:
            kind = "supplement"
        rows.append(dict(year=year, season=f"{year}/{year+1}", stage=stage,
                         kind=kind, label=label, url=url, source_url=source_url,
                         original_url=raw_url, repaired_malformed_href=(url != raw_url)))
    return rows


def results_metadata(html):
    soup = BeautifulSoup(html, "html.parser")
    year, stage = None, None
    cutoffs, assets = [], []
    for tag in soup.find_all(["h2", "h3", "span", "table", "a"]):
        text = tidy(tag.get_text(" ", strip=True))
        if tag.name in ("h2", "h3"):
            match = re.search(r"(20\d{2})\s*[-–/]\s*(20\d{2})\s*учебный", text)
            if match and tag.name == "h2":
                year, stage = int(match[1]), None
        if tag.name in ("h2", "h3", "span") and len(text) < 65:
            parsed = stage_id(text)
            if parsed:
                stage = parsed
        if year is None:
            continue
        if tag.name == "table" and "Мин. кол-во баллов" in text and stage in ("stage1", "stage2"):
            for tr in tag.find_all("tr"):
                cells = [tidy(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
                if len(cells) != 3 or not re.fullmatch(r"(?:9|10|11) класс", cells[0]):
                    continue
                number = lambda s: float(re.search(r"\d+(?:[,.]\d+)?", s)[0].replace(",", "."))
                cutoffs.append(dict(year=year, season=f"{year}/{year+1}", stage=stage,
                    grade=int(cells[0].split()[0]), winner_cutoff=number(cells[1]),
                    cutoff=number(cells[2]), source_url=BASE + "/result",
                    target_definition="minimum second-stage prizewinner score qualifies for final" if stage == "stage2" else "stage1 prizewinner threshold"))
        if tag.name == "a" and "Результат" in text.capitalize() and tag.get("href"):
            url = urljoin(BASE, tag["href"])
            if re.search(r"\.(xlsx?|pdf)$", url, re.I) or "/mirror/pubs/share/" in url:
                assets.append(dict(year=year, season=f"{year}/{year+1}", stage=stage,
                                   kind="results", label=text, url=url, source_url=BASE + "/result"))
    return cutoffs, assets


def download_asset(row):
    row = dict(row)
    url = row["url"]
    host = urlparse(url).hostname or ""
    if host not in ("dano.hse.ru", "www.hse.ru"):
        row["status"] = "external_reference"
        return row
    name = urlparse(url).path.rstrip("/").split("/")[-1]
    if not re.search(r"\.[a-zA-Z0-9]{2,5}$", name):
        name += ".bin"
    path = RAW / ("results" if row["kind"] == "results" else "archive") / str(row["year"]) / f"{row['stage']}_{name}"
    if path.suffix == ".bin" and path.with_suffix(".pdf").exists():
        path = path.with_suffix(".pdf")
    try:
        content = fetch(url, path)
        if content.startswith(b"%PDF") and path.suffix != ".pdf":
            new_path = path.with_suffix(".pdf")
            path.replace(new_path)
            path = new_path
        elif content.startswith(b"PK") and row["kind"] == "results" and path.suffix != ".xlsx":
            new_path = path.with_suffix(".xlsx")
            path.replace(new_path)
            path = new_path
        row.update(status="downloaded", path=str(path.relative_to(ROOT)), bytes=len(content),
                   sha256=hashlib.sha256(content).hexdigest(), retrieved_utc=datetime.now(timezone.utc).isoformat())
        if not content.startswith((b"%PDF", b"PK")) and b"<html" in content[:2000].lower():
            row["status"] = "html_reference"
    except Exception as exc:
        row.update(status="failed", error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-pages", action="store_true")
    args = parser.parse_args()
    pages = {}
    for key, url in {"past_tasks": BASE + "/library/past_tasks", "result": BASE + "/result", "demo": BASE + "/demo", "second": BASE + "/second"}.items():
        path = RAW / "pages" / f"{key}.html"
        if args.refresh_pages:
            path.unlink(missing_ok=True)
        pages[key] = fetch(url, path).decode("utf-8")
    rows = archive_assets(pages["past_tasks"], BASE + "/library/past_tasks")
    rows += archive_assets(pages["demo"], BASE + "/demo", 2022)
    cutoffs, result_rows = results_metadata(pages["result"])
    rows += result_rows
    # Include linked official project data pages and their downloadable attachments.
    for row in list(rows):
        if row["kind"] == "data_page" and urlparse(row["url"]).hostname == "dano.hse.ru":
            path = RAW / "pages" / (urlparse(row["url"]).path.strip("/").replace("/", "_") + ".html")
            try:
                content = fetch(row["url"], path).decode("utf-8")
                for link in BeautifulSoup(content, "html.parser").find_all("a", href=True):
                    url = urljoin(row["url"], link["href"])
                    if re.search(r"\.(pdf|xlsx?|csv|zip)(?:\?|$)", url, re.I) or ("/mirror/pubs/share/" in url and not link.find("img")):
                        rows.append(dict(year=row["year"], season=row["season"], stage=row["stage"], kind="project_attachment",
                                         label=tidy(link.get_text(" ")), url=url, source_url=row["url"]))
            except Exception as exc:
                print("data page:", row["url"], str(exc), flush=True)
    distinct = {}
    for row in rows:
        distinct.setdefault((row["url"], row["year"], row["stage"]), row)
    rows = list(distinct.values())
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_asset, row) for row in rows]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(row["year"], row["stage"], row["kind"], row["status"], row["url"], flush=True)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cutoffs).drop_duplicates(["year", "grade", "stage"]).sort_values(["year", "stage", "grade"]).to_csv(out / "cutoffs_all_stages.csv", index=False)
    pd.DataFrame([c for c in cutoffs if c["stage"] == "stage2" and c["year"] < 2026]).drop_duplicates(["year", "grade"]).sort_values(["year", "grade"]).to_csv(out / "cutoffs.csv", index=False)
    completed.sort(key=lambda r: (r["year"], str(r["stage"]), r["url"]))
    (out / "source_manifest.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n")
    print("Manifest:", len(completed), "assets; historical stage2 cutoffs:", sum(c["stage"] == "stage2" and c["year"] < 2026 for c in cutoffs), flush=True)


if __name__ == "__main__":
    main()
