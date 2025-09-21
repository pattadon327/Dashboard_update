import argparse, json, re, time, sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils import slugify

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TrafficCounter/1.0)"}

def discover_cameras(index_url: str):
    """
    Best-effort scraper that looks for video/iframe tags and HLS (.m3u8) or MP4 sources.
    Depending on site structure (JS-rendered), this may return none and require manual config.
    """
    cams = []
    r = requests.get(index_url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Look for <source> and <video>
    for vid in soup.find_all(["video", "source"]):
        src = vid.get("src") or vid.get("data-src")
        if not src:
            continue
        full = urljoin(index_url, src)
        if ".m3u8" in full or full.endswith(".mp4"):
            name = vid.get("id") or vid.get("title") or full.split("/")[-1]
            cams.append({"name": name, "slug": slugify(name), "url": full, "enabled": True})

    # Look for iframes, then fetch inside for streams
    for ifr in soup.find_all("iframe"):
        src = ifr.get("src")
        if not src:
            continue
        iframe_url = urljoin(index_url, src)
        try:
            r2 = requests.get(iframe_url, headers=HEADERS, timeout=15)
            r2.raise_for_status()
            s2 = BeautifulSoup(r2.text, "html.parser")
            for tag in s2.find_all(["source", "video"]):
                src2 = tag.get("src") or tag.get("data-src")
                if not src2:
                    continue
                full2 = urljoin(iframe_url, src2)
                if ".m3u8" in full2 or full2.endswith(".mp4"):
                    name = tag.get("id") or tag.get("title") or full2.split("/")[-1]
                    cams.append({"name": name, "slug": slugify(name), "url": full2, "enabled": True})
        except Exception:
            continue

    # Deduplicate by URL
    dedup = {}
    for c in cams:
        dedup[c["url"]] = c
    return list(dedup.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="http://www.bmatraffic.com/index.aspx", help="BMATraffic index page")
    ap.add_argument("--save", default="config/cameras.json")
    args = ap.parse_args()

    cams = discover_cameras(args.index)
    if not cams:
        print("No streams auto-discovered. You may need to fill config/cameras.json manually.", file=sys.stderr)

    out = {"timezone": "Asia/Bangkok", "cameras": cams}
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cams)} cameras to {args.save}")

if __name__ == "__main__":
    main()
