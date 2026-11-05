"""Build anonymous empirical difficulty labels and statement-only text records."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed"

# Verified against PDF point values and workbook scoring headers. A physical
# problem can be split into several automatically scored workbook columns.
ITEM_ALIGNMENT = {
    2021: [[i] for i in range(1, 26)],
    2022: [[1], [2], [3], [4], [5], [6], [7], [8, 9], [10], [11],
           [12], [13], [14, 15, 16], [17], [18]],
    2023: [[1], [2], [3], [4], [5], [6], [7, 8, 9, 10], [11], [12],
           [13], [14], [15], [16], [17], [18], [19, 20, 21, 22, 23, 24, 25], [26]],
    2024: [[i] for i in range(1, 21)],
    2025: [[i] for i in range(1, 21)],
}


def normalized_name(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е")
    return " ".join(re.findall(r"[а-яa-z0-9]+", text))


def number(value):
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or str(value).strip() in ("", "-", "—"):
        return None
    try:
        return float(str(value).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


def nonempty_rows(sheet):
    # Workbooks declare Excel's entire million-row/16384-column grid as used.
    # Bound the useful columns and stop after a long blank run, without loading
    # an empty spreadsheet grid into memory.
    blanks = 0
    for row in sheet.iter_rows(max_row=30000, max_col=64, values_only=True):
        if all(v is None for v in row):
            blanks += 1
            if blanks >= 200:
                break
        else:
            blanks = 0
        yield list(row)


def empirical_labels():
    result, audit, legacy = [], [], []
    for year in range(2021, 2026):
        glob = "None_*" if year == 2021 else "stage2_*"
        path = next((ROOT / f"data/raw/results/{year}").glob(glob))
        workbook = load_workbook(path, read_only=True, data_only=True)
        overall = list(nonempty_rows(workbook.worksheets[0]))
        header_index = next(i for i, r in enumerate(overall) if any(str(v or "").startswith(("Класс", "Фамилия")) for v in r))
        header = overall[header_index]
        if year == 2021:
            namecols, gradecol, totalcol, statuscol = [0], 1, 6, 8
        elif year == 2022:
            namecols, gradecol, totalcol, statuscol = [0], 2, 6, 7
        elif year == 2023:
            namecols, gradecol, totalcol, statuscol = [0, 1], 3, 7, 8
        else:
            namecols, gradecol, totalcol, statuscol = [0, 1, 2], 4, 8, 9
        lookup = defaultdict(set)
        invitations = defaultdict(list)
        eligible = 0
        for row in overall[header_index+1:]:
            grade_match = re.search(r"(?:^|\D)(9|10|11)(?:\D|$)", str(row[gradecol] or ""))
            if not grade_match or number(row[totalcol]) is None:
                continue
            if "дисквал" in str(row[statuscol]).lower():
                continue
            grade = int(grade_match[1])
            name = normalized_name(" ".join(str(row[c] or "") for c in namecols))
            if not name:
                continue
            lookup[name].add(grade)
            eligible += 1
            if year == 2021 and number(row[7]) is not None:
                invitations[grade].append((number(row[7]), "Приглашен" in str(row[8])))
        if year == 2021:
            for grade, vals in invitations.items():
                legacy.append(dict(year=2021, season="2021/2022", grade=grade,
                    max_not_invited=max(score for score, inv in vals if not inv),
                    min_invited=min(score for score, inv in vals if inv),
                    maximum_score=225, definition="observed first+second-stage total invitation boundary; incompatible with 100-point second-stage target",
                    source_url="https://dano.hse.ru/mirror/pubs/share/841534365.xlsx"))
        details = workbook["Второй (отборочный) этап"] if year == 2021 else workbook.worksheets[1]
        rows = list(nonempty_rows(details))
        detail_header = rows[1]
        columns = []
        for col, value in enumerate(detail_header):
            text = str(value or "")
            match = re.match(r"\s*(?:В\.\s*)?(\d+)\s*(?:\nMax\s*=\s*|/)([\d,.]+)", text)
            if match and float(match[2].replace(",", ".")) > 0:
                columns.append((col, int(match[1]), float(match[2].replace(",", "."))))
        if not columns:
            raise ValueError(f"No item maximum-score headers found: {path}")
        alignment = ITEM_ALIGNMENT[year]
        assert sorted(i for group in alignment for i in group) == sorted(c[1] for c in columns)
        total_col = 2 if year >= 2024 else 1
        detail_namecols = [0, 1] if year >= 2024 else [0]
        scores = defaultdict(list)
        counts = Counter()
        seen = set()
        for row in rows[2:]:
            if not row[0] or number(row[total_col]) is None:
                continue
            counts["detail_numeric_total_rows"] += 1
            name = normalized_name(" ".join(str(row[c] or "") for c in detail_namecols))
            grades = lookup.get(name, set())
            if len(grades) != 1:
                counts["unmatched_or_ambiguous_grade"] += 1
                continue
            grade = next(iter(grades))
            key = (name, grade, tuple(str(row[c]) for c, _, _ in columns))
            if key in seen:
                counts["duplicate_detail_rows_removed"] += 1
                continue
            seen.add(key)
            item_scores = [number(row[c]) or 0.0 for c, _, _ in columns]
            total_diff = abs(sum(item_scores) - number(row[total_col]))
            if total_diff > 0.05:
                counts["item_total_mismatch_excluded"] += 1
                continue
            if any(value < -1e-8 or value > maximum + 1e-8 for value, (_, _, maximum) in zip(item_scores, columns)):
                counts["invalid_item_score_excluded"] += 1
                continue
            counts[f"matched_grade_{grade}"] += 1
            by_item = {item: (value, maximum) for value, (_, item, maximum) in zip(item_scores, columns)}
            for item, group in enumerate(alignment, 1):
                value = sum(by_item[q][0] for q in group)
                maximum = sum(by_item[q][1] for q in group)
                scores[(grade, item, maximum)].append(value / maximum)
        for (grade, item, maximum), values in scores.items():
            arr = np.asarray(values)
            mean = float(arr.mean())
            result.append(dict(year=year, grade=grade, item=item,
                max_points=maximum, n=len(arr), mean_fraction=mean,
                difficulty=float(np.clip(100 * (1 - mean), .001, 99.999)),
                difficulty_standard_error=float(100 * arr.std(ddof=1) / np.sqrt(len(arr))),
                label_kind="100*(1-mean partial-credit fraction), conditional on observed stage2 participants",
                source_path=str(path.relative_to(ROOT))))
        audit.append(dict(year=year, overall_numeric_eligible_rows=eligible,
            ambiguous_name_keys=sum(len(g)>1 for g in lookup.values()), scoring_column_count=len(columns),
            physical_problem_count=len(alignment),
            item_max_points_sum=sum(c[2] for c in columns), **counts))
    pd.DataFrame(result).sort_values(["year", "grade", "item"]).to_csv(OUT / "difficulty_labels.csv", index=False)
    pd.DataFrame(legacy).sort_values("grade").to_csv(OUT / "legacy_2021_invitation_bounds.csv", index=False)
    (OUT / "workbook_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (OUT / "item_alignment.json").write_text(json.dumps(ITEM_ALIGNMENT, indent=2)+"\n")
    return pd.DataFrame(result)


def extract_pdf(path):
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                          capture_output=True, text=True).stdout


def split_statements(text, expected=None, current=False, legacy=False):
    text = unicodedata.normalize("NFKC", text).replace("\r", "")
    text = re.sub(r"(?m)^\s*\d+\s*/\s*\d+\s*$", "", text)
    # 2022 files contain both an answered full paper and an incomplete clean
    # copy. Keep the full paper; clean copies are used below where available.
    clean_copy = re.search(r"(?m)^\s*Вариант\s+\d+\s*\(без ответов\)\s*$", text)
    if clean_copy:
        text = text[:clean_copy.start()]
    explicit = r"(?m)^\s*(?:Задача|Вопрос)\s+(\d{1,2})(?:\s*\([^\n]*\))?\s*$"
    pattern = explicit if re.search(explicit, text) else r"(?m)^\s*(\d{1,2})\.\s+(?=[А-ЯЁа-яёA-Za-z(\[])"
    candidates = list(re.finditer(pattern, text))
    chosen, nxt = [], 1
    for match in candidates:
        if int(match[1]) == nxt:
            chosen.append(match)
            nxt += 1
    if expected is not None and len(chosen) != expected:
        raise ValueError(f"Expected {expected} statements, found {len(chosen)}; inspect PDF rather than silently misalign labels")
    if len(chosen) < 10:
        raise ValueError(f"Only {len(chosen)} statements extracted")
    records = []
    for j, match in enumerate(chosen):
        end = chosen[j+1].start() if j+1 < len(chosen) else len(text)
        statement = text[match.end():end]
        statement = re.split(r"(?mi)^\s*(?:Ответы|Решения|Критерии оценивания)\s*[:.]?\s*$", statement)[0]
        # Several official archive variants include correct answers immediately
        # after each question. They must never enter statement features.
        if legacy:
            statement = re.split(r"(?mi)^\s*(?:Ответ(?:ы)?|Решение)\s*[:—-]", statement)[0]
        else:
            statement = re.sub(r"(?mi)^\s*(?:Ответ(?:ы)?|Решение)\s*[:—-][^\n]*", "", statement)
        statement = re.sub(r"(?mi)^([^\n=]+)=([^\n]*\(Ответ:[^\n]*)", r"\1= [blank]", statement)
        statement = re.sub(r"\(Ответ:[^)]*\)", "", statement, flags=re.I)
        statement = re.sub(r"\(\s*\+\s*\)", "", statement)
        statement = re.sub(r"(?m)^\s*\([−-]?\d+[,.]\d+\)\s*$", "", statement)
        statement = re.sub(r"\s+", " ", statement).strip()
        statement = re.sub(r"^\(\s*\d+(?:[,.]\d+)?\s+балл[а-я]*\s*\)\s*", "", statement)
        canonical = " ".join(re.findall(r"[а-яa-z]+", statement.lower().replace("ё", "е")))
        records.append(dict(item=j+1, text=statement,
            text_hash=hashlib.sha256(statement.encode()).hexdigest(),
            family_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            text_characters=len(statement), page_start=text[:match.start()].count("\f")+1,
            page_end=text[:end].count("\f")+1))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = empirical_labels()
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    documents = [r for r in manifest if r["stage"] == "stage2" and r["kind"] == "problem" and r["status"] == "downloaded"]
    records, audit = [], []
    for doc in sorted(documents, key=lambda r: (r["year"], r["label"])):
        year = int(doc["year"])
        variant = int(re.search(r"\d+", doc["label"])[0])
        expected = int(labels[labels.year == year].item.max())
        full_text = extract_pdf(ROOT / doc["path"])
        statements = split_statements(full_text, expected, legacy=(year == 2021))
        if year == 2022:
            clean_match = re.search(r"(?m)^\s*Вариант\s+\d+\s*\(без ответов\)\s*$", full_text)
            if clean_match:
                clean_statements = split_statements(full_text[clean_match.end():])
                title = lambda r: re.sub(r"\s+", " ", r["text"].split("(")[0]).strip()
                clean_by_title = {title(r): r for r in clean_statements}
                for j, original in enumerate(statements):
                    if title(original) in clean_by_title:
                        replacement = dict(clean_by_title[title(original)])
                        replacement["item"] = original["item"]
                        statements[j] = replacement
        for row in statements:
            records.append(dict(problem_id=f"{year}_v{variant}_q{row['item']}",
                year=year, variant=variant, source_path=doc["path"], source_url=doc["url"], **row))
        audit.append(dict(year=year, variant=variant, n_problems=len(statements), pdf_path=doc["path"],
                          sha256=doc["sha256"]))
    current_path = ROOT / "data/raw/current/2026_second_stage_user.pdf"
    statements = split_statements(extract_pdf(current_path), 30, current=True)
    for row in statements:
        records.append(dict(problem_id=f"2026_v1_q{row['item']}", year=2026, variant=1,
            source_path=str(current_path.relative_to(ROOT)), source_url="user-supplied PDF", **row))
    audit.append(dict(year=2026, variant=1, n_problems=len(statements), pdf_path=str(current_path.relative_to(ROOT)),
        sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
        limitations="No item point weights or attached task data in supplied printout. Some formula glyphs/diagrams are images and absent from pdftotext. Prediction assumes total score 100."))
    (OUT / "problem_texts.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records)+"\n")
    metadata = [{k: v for k, v in r.items() if k != "text"} for r in records]
    pd.DataFrame(metadata).to_csv(OUT / "problems_metadata.csv", index=False)
    (OUT / "pdf_extraction_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    print("Empirical labels:", len(labels), "statement variants:", len(records), flush=True)
    print(pd.DataFrame(audit)[["year", "variant", "n_problems"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
