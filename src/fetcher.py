import sqlite3, json, requests, time, logging
from pathlib import Path
from config import (
    INDIAN_KANOON_API_KEY, INDIAN_KANOON_BASE_URL,
    DB_PATH, START_WITH_N_CASES
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── SCHEMA VERIFIED: SnehaDeshmukh/IndianBailJudgments-1200 ───────────────
# Fields: facts, judgment_reason, summary, ipc_sections, court, date,
#         bail_outcome, crime_type, case_title, judge, accused_name, etc.
# Text = combined facts + judgment_reason + summary (~870 chars avg)
DATASET_NAME = "SnehaDeshmukh/IndianBailJudgments-1200"
TEXT_FIELDS = ["facts", "judgment_reason", "summary"]  # combined into raw_text
COURT_FIELD = "court"
DATE_FIELD = "date"
VERDICT_FIELD = "bail_outcome"


def validate_schema(row: dict) -> tuple:
    """
    v3.1: Validate first row before processing all cases.
    Returns (is_valid: bool, reason: str)
    """
    for field in TEXT_FIELDS:
        if field not in row:
            return False, (
                f"Expected field '{field}' not found in row. "
                f"Available fields: {list(row.keys())}. "
                f"Dataset schema may have changed."
            )

    # Combine text fields and check length
    combined = " ".join(str(row.get(f, "")) for f in TEXT_FIELDS)
    if len(combined.strip()) < 50:
        return False, (
            f"Combined text fields too short ({len(combined)} chars). "
            f"Fields: {TEXT_FIELDS}"
        )
    return True, "ok"


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            court TEXT,
            date TEXT,
            raw_text TEXT,
            source TEXT,
            meta TEXT
        )
    """)
    conn.commit()
    return conn


def fetch_from_huggingface(max_cases=500):
    from datasets import load_dataset

    log.info(f"Loading {max_cases} diverse cases from {DATASET_NAME}...")
    ds = load_dataset(
        DATASET_NAME,
        split="train",
    )
    
    # Stratified sampling: shuffle to ensure diversity instead of just top N
    if len(ds) > max_cases:
        ds = ds.shuffle(seed=42).select(range(max_cases))
    else:
        log.warning(f"Dataset has {len(ds)} cases, requesting {max_cases}. Taking all.")
        ds = ds.shuffle(seed=42)

    if len(ds) == 0:
        raise ValueError("Dataset returned 0 rows.")

    first_row = dict(ds[0])
    valid, reason = validate_schema(first_row)
    if not valid:
        raise ValueError(
            f"SCHEMA VALIDATION FAILED: {reason}\n"
            f"Run the Phase 2.1 schema verification snippet first."
        )

    log.info(
        f"Schema valid. TEXT_FIELDS={TEXT_FIELDS}. "
        f"Processing {len(ds)} rows..."
    )

    empty_text_count = 0
    cases = []

    for i, row in enumerate(ds):
        # Combine text fields: facts + judgment_reason + summary
        text_parts = []
        for field in TEXT_FIELDS:
            val = str(row.get(field, "")).strip()
            if val and val.lower() != "none":
                text_parts.append(val)
        combined_text = " ".join(text_parts)

        if len(combined_text.strip()) < 50:
            empty_text_count += 1
            continue

        # Extract IPC sections from the dataset (already parsed!)
        ipc_raw = row.get("ipc_sections", "[]")
        try:
            if isinstance(ipc_raw, str):
                ipc_sections = json.loads(ipc_raw.replace("'", '"'))
            elif isinstance(ipc_raw, list):
                ipc_sections = ipc_raw
            else:
                ipc_sections = []
        except (json.JSONDecodeError, Exception):
            ipc_sections = []

        case_id = str(row.get("case_id", f"bail_{i}"))

        cases.append({
            "id": f"hf_{case_id}",
            "court": str(row.get(COURT_FIELD, "unknown")),
            "date": str(row.get(DATE_FIELD, "")),
            "raw_text": combined_text,
            "source": "huggingface",
            "meta": json.dumps({
                "length": len(combined_text),
                "row_index": i,
                "dataset": DATASET_NAME,
                "case_title": str(row.get("case_title", "")),
                "bail_outcome": str(row.get(VERDICT_FIELD, "")),
                "crime_type": str(row.get("crime_type", "")),
                "ipc_sections": ipc_sections,
                "judge": str(row.get("judge", "")),
                "accused_name": str(row.get("accused_name", "")),
                "bail_type": str(row.get("bail_type", "")),
            })
        })

    log.info(
        f"Loaded {len(cases)} valid cases. "
        f"{empty_text_count} skipped (empty text)."
    )

    if empty_text_count > max_cases * 0.3:
        log.warning(
            f"WARNING: {empty_text_count}/{max_cases} rows had empty text. "
            f"TEXT_FIELDS={TEXT_FIELDS} may be wrong."
        )

    return cases


def fetch_from_indian_kanoon(query, pages=5):
    if not INDIAN_KANOON_API_KEY:
        log.info("No Indian Kanoon API key. Skipping.")
        return []
    headers = {"Authorization": f"Token {INDIAN_KANOON_API_KEY}"}
    cases = []
    for page in range(1, pages + 1):
        try:
            r = requests.post(
                f"{INDIAN_KANOON_BASE_URL}/search/",
                data={"formInput": query, "pagenum": page},
                headers=headers,
                timeout=30
            )
            r.raise_for_status()
            for doc in r.json().get("docs", []):
                cases.append({
                    "id": str(doc.get("tid", "")),
                    "court": doc.get("court", "unknown"),
                    "date": doc.get("publishdate", ""),
                    "raw_text": doc.get("doc", ""),
                    "source": "indiankanoon",
                    "meta": json.dumps(doc)
                })
            time.sleep(1)
        except Exception as e:
            log.error(f"Page {page} error: {e}")
    log.info(f"Fetched {len(cases)} cases for '{query}'.")
    return cases


def save_cases_to_db(cases, conn):
    inserted = 0
    for c in cases:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?)",
                (c["id"], c["court"], c["date"],
                 c["raw_text"], c["source"], c["meta"])
            )
            inserted += 1
        except Exception as e:
            log.error(f"Insert error {c['id']}: {e}")
    conn.commit()
    log.info(f"Saved {inserted} new cases to DB.")
    return inserted


def get_case_count(conn):
    return conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]


if __name__ == "__main__":
    conn = init_database()
    hf_cases = fetch_from_huggingface(max_cases=START_WITH_N_CASES)
    save_cases_to_db(hf_cases, conn)
    total = get_case_count(conn)
    print(f"\nTotal cases in DB: {total}")

    if INDIAN_KANOON_API_KEY:
        queries = [
            "IPC 302 murder Supreme Court",
            "IPC 376 rape acquittal",
            "IPC 420 fraud cheating",
        ]
        for q in queries:
            ik_cases = fetch_from_indian_kanoon(q, pages=5)
            save_cases_to_db(ik_cases, conn)
        print(f"Total after Indian Kanoon: {get_case_count(conn)}")

    conn.close()
