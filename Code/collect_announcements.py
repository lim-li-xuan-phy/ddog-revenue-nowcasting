'''
Scrapes Datadog press releases from their newsroom website.
Classifies each headline into TRUE or FALSE for product_release using keyword rules.
Saves into product_announcements.csv in the Data directory. 
'''

import csv
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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

END_DATE = datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc) # End of Q2 2026
DATADOG_PRESS_HTML_URL = "https://www.datadoghq.com/about/latest-news/press-releases/"
DATADOG_PRESS_RSS_URL = "https://www.datadoghq.com/about/latest-news/press-releases/index.xml"
DATADOG_SITEMAP_URL = "https://www.datadoghq.com/en/sitemap.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def classify_product_release(headline: str) -> str:
    '''
    Hard-classifies a headline into TRUE or FALSE for product_release using keywords.
    '''
    h = headline.lower().strip()

    true_keywords = ['enhance', 'launches', 'to launch', 'announce', 'introduce', 'release', 'expand', 'feature', 'add', 'available', 'extend']
    false_keywords = ['centre', 'center']

    if any(kw in h for kw in false_keywords):
        return "FALSE"

    if any(kw in h for kw in true_keywords):
        return "TRUE"

    return "FALSE"


def extract_announcement_details(url: str):
    '''
    Retrieves a press release HTML page to extract title and published_time meta tag.
    Returns tuple: (headline, ts) or None.
    '''
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            meta_date = soup.find("meta", property="article:published_time") or soup.find("meta", name="date") or soup.find("meta", property="og:published_time")
            date_str = meta_date["content"] if meta_date and meta_date.get("content") else ""

            h1 = soup.find("h1") or soup.find("title")
            headline = h1.get_text(strip=True) if h1 else ""
            headline = re.sub(r'\s*\|\s*Datadog.*$', '', headline).strip()

            if headline and date_str:
                # Handle dates like '2012-11-20 14:42:00 +0000 UTC'
                clean_date = date_str[:19].replace("T", " ")
                ts = datetime.strptime(clean_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                return headline, ts
    except Exception:
        pass
    return None


def deduplicate_by_timestamp(records: list) -> list:
    '''
    Ensures every record headline is mapped to a unique timestamp `ts`.
    If two records share the exact same ts, increments subsequent records by 1 second.
    '''
    records.sort(key=lambda x: x[0])
    deduped = []
    seen_timestamps = set()

    for ts, headline, product_release in records:
        current_ts = ts
        while current_ts in seen_timestamps:
            current_ts += timedelta(seconds=1)
        seen_timestamps.add(current_ts)
        deduped.append((current_ts, headline, product_release))

    return deduped


def load_existing_announcements(target_csv: Path) -> tuple[list, set, datetime | None]:
    '''
    Loads existing announcements from target_csv.
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
                is_rel = row.get("product_release", "FALSE").strip().upper()

                if not ts_str or not headline:
                    continue

                try:
                    dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                records.append((dt, headline, is_rel))
                seen_headlines.add(headline.lower().strip())

        if records:
            records.sort(key=lambda x: x[0])
            latest_ts = records[-1][0]
    except Exception as e:
        log.warning(f"Could not read existing file {target_csv.name}: {e}")
        return [], set(), None

    return records, seen_headlines, latest_ts


def scrape_datadog_announcements(latest_ts: datetime = None, seen_headlines: set = None, max_ts: datetime = None) -> list:
    '''
    Scrapes announcements from Datadog's newsroom website.
    If latest_ts is provided, only retrieves records where latest_ts < ts <= max_ts.
    Returns list of tuples: (timestamp_utc, headline, product_release)
    '''
    if max_ts is None:
        max_ts = datetime.now(timezone.utc)
    if seen_headlines is None:
        seen_headlines = set()

    announcements_dict = {}  # headline -> (timestamp_utc, headline, product_release)

    # RSS Feed items (recent announcements)
    try:
        log.info(f"Retrieving Datadog press releases RSS feed from {DATADOG_PRESS_RSS_URL}...")
        resp = requests.get(DATADOG_PRESS_RSS_URL, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall("./channel/item"):
                title_el = item.find("title")
                pubDate_el = item.find("pubDate")

                if title_el is not None and title_el.text:
                    headline = title_el.text.strip()
                    ts = None
                    if pubDate_el is not None and pubDate_el.text:
                        try:
                            ts = parsedate_to_datetime(pubDate_el.text.strip())
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            else:
                                ts = ts.astimezone(timezone.utc)
                        except Exception:
                            pass

                    if ts is None:
                        ts = datetime.now(timezone.utc)

                    # Filter by date bounds and duplicate headline check
                    if latest_ts is not None and ts <= latest_ts:
                        continue
                    if ts > max_ts:
                        continue
                    if headline.lower().strip() in seen_headlines:
                        continue

                    is_rel = classify_product_release(headline)
                    announcements_dict[headline] = (ts, headline, is_rel)
    except Exception as e:
        log.error(f"Error retrieving RSS feed: {e}")

    # Sitemap index (used for initial full collection or discovering articles not in RSS)
    # If this is an incremental update and RSS already provided recent items, we only check unknown URLs in sitemap
    try:
        log.info(f"Retrieving Datadog sitemap...")
        resp_sm = requests.get(DATADOG_SITEMAP_URL, headers=HEADERS, timeout=15)
        if resp_sm.status_code == 200:
            sm_root = ET.fromstring(resp_sm.content)
            announcement_urls = [
                l.text for l in sm_root.findall(".//{*}loc")
                if l.text and "/about/latest-news/press-releases/" in l.text and l.text.rstrip('/') != "https://www.datadoghq.com/about/latest-news/press-releases"
            ]
            log.info(f"Found {len(announcement_urls)} pages in Datadog Newsroom sitemap.")

            # Build set of normalized strings from seen headlines
            seen_normalized = {re.sub(r'[^a-z0-9]', '', h) for h in seen_headlines}

            # Filter out URLs whose slug is already seen in existing records
            urls_to_retrieve = []
            for url in announcement_urls:
                raw_slug = url.rstrip("/").split("/")[-1]
                norm_slug = re.sub(r'[^a-z0-9]', '', raw_slug)
                if norm_slug in seen_normalized:
                    continue
                # Also check if any existing normalized headline contains significant portion of slug
                if len(norm_slug) > 10 and any(norm_slug[:20] in h for h in seen_normalized):
                    continue
                urls_to_retrieve.append(url)

            if urls_to_retrieve:
                log.info(f"Scraping press release details for {len(urls_to_retrieve)} URLs...")
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = {executor.submit(extract_announcement_details, url): url for url in urls_to_retrieve}
                    for future in as_completed(futures):
                        res = future.result()
                        if res:
                            headline, ts = res
                            if latest_ts is not None and ts <= latest_ts:
                                continue
                            if ts > max_ts:
                                continue
                            if headline.lower().strip() in seen_headlines:
                                continue
                            if headline not in announcements_dict:
                                is_rel = classify_product_release(headline)
                                announcements_dict[headline] = (ts, headline, is_rel)
    except Exception as e:
        log.error(f"Error scraping press release details from Datadog sitemap: {e}")

    records = list(announcements_dict.values())
    return records


def save_announcements_to_csv(target_csv: Path, rows: list) -> int:
    '''
    Saves/upserts announcement records into a CSV file.
    '''
    if not rows:
        log.warning("No announcement records to save.")
        return 0

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "headline", "product_release"])
        for item in rows:
            ts = item[0]
            headline = item[1]
            product_release = item[2]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S+00:00") if isinstance(ts, datetime) else str(ts)
            writer.writerow([ts_str, headline, product_release])

    return len(rows)


def run():
    current_ts = datetime.now(timezone.utc)
    target_csv = DATA_DIR / "announcements.csv"
    log.info(f"Running Datadog announcements collector (current timestamp: {current_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}) ...")

    existing_records, seen_headlines, latest_ts = load_existing_announcements(target_csv)

    if existing_records and latest_ts:
        log.info(f"Found existing dataset with {len(existing_records)} announcements up to {latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}.")
        log.info(f"Checking for new announcements between {latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} and {current_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} ...")
        new_records = scrape_datadog_announcements(latest_ts=latest_ts, seen_headlines=seen_headlines, max_ts=current_ts)

        if new_records:
            log.info(f"Found {len(new_records)} new announcements. Merging with existing dataset...")
            combined = deduplicate_by_timestamp(existing_records + new_records)
            n = save_announcements_to_csv(target_csv, combined)
            log.info(f"Added {len(new_records)} new records. Saved total of {n} records into {target_csv.name}.")
        else:
            log.info(f"Announcements in {target_csv.name} are already up to date. No new announcements found between {latest_ts.strftime('%Y-%m-%d')} and {current_ts.strftime('%Y-%m-%d')}.")
    else:
        log.info(f"No existing announcements dataset found. Scraping full history up to {current_ts.strftime('%Y-%m-%d')} ...")
        records = scrape_datadog_announcements(max_ts=current_ts)
        records = deduplicate_by_timestamp(records)
        log.info(f"Processed {len(records)} Datadog announcements.")
        n = save_announcements_to_csv(target_csv, records)
        log.info(f"Saved {n} records into {target_csv.name}.")

    log.info("Datadog announcements collection & classification completed.")


if __name__ == "__main__":
    run()

