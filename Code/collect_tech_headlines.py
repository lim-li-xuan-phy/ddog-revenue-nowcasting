'''
Scrapes technology-related news headlines from Wayback Machine snapshots of the CNBC technology page. For
each calendar month between START_DATE and END_DATE the Wayback Machine CDX API is queried to retrieve 
one representative archived snapshot. Headlines that do not contain the word "cloud" are filtered out.
Snapshots are deduplicated by timestamps.

Classifies headlines into 3 themes (threats_and_vulnerabilities / neutral / opportunities_for_growth)
using a SetFit NLP model trained on labelled examples.
Records the predicted probabilities of each theme and computes news_score from the theme under which
the maximum probability falls:
- If threats_and_vulnerabilities, news_score = -prob
- If opportunities_for_growth, news_score = +prob
- If neutral, left empty

Saves into DATA_DIR/tech_headlines.csv.
'''

import csv
import json
import logging
import os
import re
import socket
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Compatibility patch for transformers
import transformers.training_args
if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: os.path.join(
        ".", "runs", datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname()
    )

from setfit import SetFitModel, Trainer, TrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Base paths
CODE_DIR = Path(__file__).parent.resolve()
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"
LABELLED_DATA_PATH = DATA_DIR / "tech_headlines_labelled.csv"

# Date bounds of data to be collected
START_DATE = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc) # 1 Jan 2021
END_DATE = datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc) # 30 Jun 2026

CNBC_TECH_URL = "https://www.cnbc.com/technology/"

# Wayback Machine endpoints
WAYBACK_CDX_URL  = "http://web.archive.org/cdx/search/cdx"
WAYBACK_BASE_URL = "https://web.archive.org/web"

# Polite-crawl delay between Wayback Machine page fetches in seconds
WAYBACK_FETCH_DELAY = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Themes into which each headline will be classified
CLASSLABELS = ["threats_and_vulnerabilities", "opportunities_for_growth", "neutral"]
LABEL2ID = {c: i for i, c in enumerate(CLASSLABELS)}
ID2LABEL = {i: c for i, c in enumerate(CLASSLABELS)}


class SetFitHeadlineClassifier:
    '''
    Uses a SetFit NLP model trained on labelled cloud technology headlines to soft-classify
    headlines into: 'threats_and_vulnerabilities', 'neutral', and 'opportunities_for_growth'.
    '''

    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.labelled_csv_path = LABELLED_DATA_PATH
        self.base_model_name = base_model_name
        self.classes = CLASSLABELS
        self.model = None
        self._init_and_train_model()

    def _init_and_train_model(self):
        '''
        Trains SetFit model on labelled data from tech_headlines_labelled.csv.
        '''
        log.info(f"Loading labelled training data from {self.labelled_csv_path}...")
        if not self.labelled_csv_path.exists():
            raise FileNotFoundError(f"Labelled CSV not found at {self.labelled_csv_path}")

        df = pd.read_csv(self.labelled_csv_path)
        
        df = df[df["classlabel"].isin(self.classes)].copy()
        df["label"] = df["classlabel"].map(LABEL2ID)

        log.info(f"Training SetFit on {len(df)} labelled samples. Class distribution: {df['classlabel'].value_counts().to_dict()}")

        train_dataset = Dataset.from_dict({
            "text": df["headline"].tolist(),
            "label": df["label"].tolist(),
        })

        model_body = SentenceTransformer(self.base_model_name)
        model_head = LogisticRegression()
        self.model = SetFitModel(
            model_body=model_body,
            model_head=model_head,
            labels=self.classes,
        )

        args = TrainingArguments(
            batch_size=16,
            num_epochs=1,
            num_iterations=20,
            seed=42,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
        )
        trainer.train()
        log.info("SetFit model training complete.")

    def predict_probs(self, headlines: list[str] | str) -> list[dict[str, float]] | dict[str, float]:
        '''
        Soft-classifies one or more headlines into class probabilities.
        Returns a dictionary (or list of dictionaries) mapping class_name -> probability.
        '''
        is_single = isinstance(headlines, str)
        input_list = [headlines] if is_single else headlines

        probs = self.model.predict_probs(input_list)
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        results = []
        for p in probs:
            prob_dict = {cls_name: round(float(p[i]), 4) for i, cls_name in enumerate(self.classes)}
            results.append(prob_dict)

        return results[0] if is_single else results


def extract_headline(url: str) -> str:
    '''
    Extracts the headline from the last section of a CNBC article URL.
    '''
    m = re.search(r'/\d{4}/\d{2}/\d{2}/([^/]+)\.html', url) # http://www.../<this section>.html
    if not m:
        return ""
    words = m.group(1).split('-')
    small_words = {'a', 'an', 'the', 'in', 'on', 'of', 'to', 'for', 'and', 'with', 'at', 'by', 'vs'}
    capitalized = [w.capitalize() if i == 0 or w.lower() not in small_words else w for i, w in enumerate(words)]
    headline = ' '.join(capitalized)
    return headline.strip()


def deduplicate_by_timestamp(records: list) -> list:
    '''
    Ensures every record headline is mapped to a unique timestamp `ts`.
    If two records share the exact same ts, increments subsequent records by 1 second.
    '''
    records.sort(key=lambda x: x[0])
    deduped = []
    seen_timestamps = set()

    for item in records:
        current_ts = item[0]
        while current_ts in seen_timestamps:
            current_ts += timedelta(seconds=1)
        seen_timestamps.add(current_ts)
        deduped.append((current_ts, *item[1:]))

    return deduped


def load_existing_tech_headlines(target_csv: Path) -> tuple[list, set, datetime | None]:
    '''
    Loads existing tech headlines from target_csv.
    Returns (records, seen_headlines_set, latest_timestamp).
    '''
    if not target_csv.exists():
        return [], set(), None

    records = []
    seen_headlines = set()
    latest_ts = None

    try:
        with open(target_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp_utc", "").strip()
                headline = row.get("headline", "").strip()
                p_t = row.get("threats_and_vulnerabilities", "").strip()
                p_g = row.get("opportunities_for_growth", "").strip()
                p_n = row.get("neutral", "").strip()
                score = row.get("news_score", "").strip()

                if not ts_str or not headline:
                    continue

                try:
                    dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                p_threats = float(p_t) if p_t else 0.0
                p_growth = float(p_g) if p_g else 0.0
                p_neutral = float(p_n) if p_n else 0.0
                score_val = float(score) if score else ""

                records.append((dt, headline, p_threats, p_growth, p_neutral, score_val))
                seen_headlines.add(headline.lower().strip())

        if records:
            records.sort(key=lambda x: x[0])
            latest_ts = records[-1][0]
    except Exception as e:
        log.warning(f"Could not read existing file {target_csv.name}: {e}")
        return [], set(), None

    return records, seen_headlines, latest_ts


def get_wayback_snapshots(start_date: datetime, end_date: datetime) -> list[tuple[str, datetime]]:
    '''
    Queries the Wayback Machine CDX API for one snapshot per calendar month of
    CNBC_TECH_URL between start_date and end_date.
    '''
    snapshots = []

    # Iterate month by month
    current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current <= end_date:
        # Formats `current` as a timestamp string
        from_ts = current.strftime("%Y%m%d%H%M%S")
        # Formats last second of the month as a timestamp string
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        to_ts = (next_month - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")

        params = {
            "url":       CNBC_TECH_URL,
            "output":    "json",
            "from":      from_ts,
            "to":        to_ts,
            "fl":        "timestamp",
            "filter":    "statuscode:200",
            "limit":     1,       # one snapshot per month is taken
            "fastLatest": "true", # return the closest available snapshot quickly
        }

        try:
            r = requests.get(WAYBACK_CDX_URL, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                rows = json.loads(r.text)
                # rows[0] is the header ["timestamp"], rows[1+] are data rows
                if len(rows) > 1:
                    ts_str = rows[1][0]  # used to build the wayback snapshot url, YYYYMMDDHHmmss
                    snap_ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) # ts_str in UTC format
                    snapshots.append((ts_str, snap_ts))
                    log.debug(f"  CDX: found snapshot {ts_str} for {current.strftime('%Y-%m')}")
                else:
                    log.debug(f"  CDX: no snapshot found for {current.strftime('%Y-%m')}")
            else:
                log.warning(f"CDX API returned HTTP {r.status_code} for {current.strftime('%Y-%m')}")
        except Exception as e:
            log.warning(f"CDX API error for {current.strftime('%Y-%m')}: {e}")

        # Advance to next month
        current = next_month

    return snapshots


def parse_headlines_from_snapshot(ts_str: str, snap_ts: datetime, start_date: datetime, end_date: datetime) -> dict:
    '''
    Fetches a Wayback Machine snapshot matched to a timestamp and extracts headlines from it.
    Filters out any headlines that do not contain 'cloud'.
    Returns a dict mapping headline_text -> article_datetime_utc.
    '''
    replay_url = f"{WAYBACK_BASE_URL}/{ts_str}/{CNBC_TECH_URL}"
    result = {}

    try:
        resp = requests.get(replay_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            log.warning(f"Snapshot {ts_str}: HTTP {resp.status_code}, skipping.")
            return result

        soup = BeautifulSoup(resp.text, "html.parser")

        for div in soup.find_all(["div", "li", "section"]):
            title_el = div.find(
                class_=lambda c: c and (
                    "Card-title" in c
                    or "LatestNews-headline" in c
                    or "RiverHeadline-headline" in c
                )
            )
            if not title_el:
                continue

            headline_text = title_el.get_text(strip=True)
            if not headline_text or len(headline_text) < 15 or headline_text in result:
                continue

            # Remove technology headlines that do not contain "cloud"
            if "cloud" not in headline_text.lower():
                continue

            # Prefer a date embedded in the article URL
            a_tag = title_el if title_el.name == "a" else title_el.find("a")
            href = ""
            if a_tag:
                raw_href = a_tag.get("href", "")
                # Strip Wayback Machine prefix if present (e.g. /web/20210101.../https://...)
                m_wb = re.search(r'https?://www\.cnbc\.com(/.*)', raw_href)
                href = m_wb.group(0) if m_wb else raw_href

            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
            if date_match:
                year, month, day = map(int, date_match.groups())
                article_ts = datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)
            else:
                # Fall back to the snapshot date
                article_ts = snap_ts.replace(hour=12, minute=0, second=0, microsecond=0)

            if start_date <= article_ts <= end_date:
                result[headline_text] = article_ts

    except Exception as e:
        log.error(f"Error fetching Wayback snapshot {ts_str}: {e}")

    return result


def scrape_tech_headlines(start_date: datetime, end_date: datetime) -> list:
    '''
    Scrapes news headlines that contain "cloud" from Wayback Machine snapshots of the CNBC technology page
    between start_date and end_date.
    Returns list of (ts_utc, headline_text) tuples.
    '''
    scraped_dict = {}  # headline_text -> ts_utc

    log.info(f"Querying Wayback Machine CDX API for monthly snapshots between {start_date.strftime('%Y-%m')} and {end_date.strftime('%Y-%m')}...")
    snapshots = get_wayback_snapshots(start_date, end_date)
    log.info(f"Found {len(snapshots)} monthly snapshots.")

    for i, (ts_str, snap_ts) in enumerate(snapshots, start=1):
        log.info(f"[{i}/{len(snapshots)}] Fetching snapshot {ts_str} ({snap_ts.strftime('%Y-%m')})...")
        month_headlines = parse_headlines_from_snapshot(ts_str, snap_ts, start_date, end_date)
        new_count = 0
        for headline_text, article_ts in month_headlines.items():
            if headline_text not in scraped_dict:
                scraped_dict[headline_text] = article_ts
                new_count += 1
        log.info(f"  -> {new_count} new headlines (running total: {len(scraped_dict)})")

        # Polite delay between Wayback Machine requests
        if i < len(snapshots):
            time.sleep(WAYBACK_FETCH_DELAY)

    records = [(ts, headline) for headline, ts in scraped_dict.items()]
    return records


def compute_news_score(p_threats: float, p_growth: float, p_neutral: float):
    '''
    Computes news_score based on the maximum probability among the three classes:
    - If threats_and_vulnerabilities is max: negative probability (-p_threats)
    - If opportunities_for_growth is max: positive probability (+p_growth)
    - If neutral is max: empty string
    '''
    probs = {
        "threats_and_vulnerabilities": p_threats,
        "opportunities_for_growth": p_growth,
        "neutral": p_neutral,
    }
    max_class = max(probs, key=probs.get)
    if max_class == "threats_and_vulnerabilities":
        return round(-p_threats, 4)
    elif max_class == "opportunities_for_growth":
        return round(p_growth, 4)
    else:  # neutral is max
        return ""


def classify_headline_items(classifier: SetFitHeadlineClassifier, items: list[tuple[datetime, str]]) -> list:
    '''
    Given a list of (timestamp_utc, headline) tuples, runs SetFit classification
    and computes news_scores.
    '''
    if not items:
        return []

    headlines = [hl for _, hl in items]
    prob_dicts = classifier.predict_probs(headlines)
    if isinstance(prob_dicts, dict):
        prob_dicts = [prob_dicts]

    records = []
    for (ts, hl), probs in zip(items, prob_dicts):
        p_threats = probs["threats_and_vulnerabilities"]
        p_growth = probs["opportunities_for_growth"]
        p_neutral = probs["neutral"]
        news_score = compute_news_score(p_threats, p_growth, p_neutral)
        records.append((
            ts,
            hl,
            p_threats,
            p_growth,
            p_neutral,
            news_score,
        ))
    return records


def save_tech_headlines_to_csv(target_csv: Path, rows: list) -> int:
    '''
    Saves tech news headline records into a CSV file.
    '''
    if not rows:
        log.warning("No headline records to save.")
        return 0

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "headline", "threats_and_vulnerabilities", "opportunities_for_growth", "neutral", "news_score"])
        for item in rows:
            ts = item[0]
            headline = item[1]
            p_threats = item[2]
            p_growth = item[3]
            p_neutral = item[4]
            news_score = item[5]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S+00:00") if isinstance(ts, datetime) else str(ts)
            writer.writerow([ts_str, headline, p_threats, p_growth, p_neutral, news_score])

    return len(rows)


def run():
    current_ts = datetime.now(timezone.utc)
    target_csv = DATA_DIR / "tech_headlines.csv"
    log.info(f"Running tech headlines collector (current timestamp: {current_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}) ...")

    existing_rows, seen_headlines, latest_ts = load_existing_tech_headlines(target_csv)

    if existing_rows and latest_ts:
        log.info(f"Found existing dataset with {len(existing_rows)} tech headlines up to {latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}.")
        if latest_ts >= current_ts:
            log.info(f"Tech headlines in {target_csv.name} are already up to date. No new headlines needed.")
            return

        query_start = latest_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        log.info(f"Checking for updates between {latest_ts.strftime('%Y-%m-%d')} and {current_ts.strftime('%Y-%m-%d')} ...")
        scraped_items = scrape_tech_headlines(query_start, current_ts)

        # Filter strictly for new items
        new_items = [
            (ts, hl) for ts, hl in scraped_items
            if ts > latest_ts and ts <= current_ts and hl.lower().strip() not in seen_headlines
        ]

        if new_items:
            log.info(f"Found {len(new_items)} new headlines. Training/loading SetFit classifier...")
            classifier = SetFitHeadlineClassifier()
            new_classified_rows = classify_headline_items(classifier, new_items)
            combined = deduplicate_by_timestamp(existing_rows + new_classified_rows)
            n = save_tech_headlines_to_csv(target_csv, combined)
            log.info(f"Added {len(new_classified_rows)} new classified headlines. Saved total of {n} records into {target_csv.name}.")
        else:
            log.info(f"Tech headlines in {target_csv.name} are already up to date. No new headlines found between {latest_ts.strftime('%Y-%m-%d')} and {current_ts.strftime('%Y-%m-%d')}.")
    else:
        log.info(f"No existing tech headlines dataset found. Scraping full history from {START_DATE.strftime('%Y-%m-%d')} to {current_ts.strftime('%Y-%m-%d')} ...")
        scraped_items = scrape_tech_headlines(START_DATE, current_ts)
        cloud_items = [(ts, hl) for ts, hl in scraped_items if "cloud" in hl.lower()]
        if cloud_items:
            classifier = SetFitHeadlineClassifier()
            classified_rows = classify_headline_items(classifier, cloud_items)
            records = deduplicate_by_timestamp(classified_rows)
            n = save_tech_headlines_to_csv(target_csv, records)
            log.info(f"Saved {n} records into {target_csv.name}.")
        else:
            log.info("No tech headlines found.")

    log.info("Tech headlines collection & classification completed.")


if __name__ == "__main__":
    run()

