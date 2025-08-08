#!/usr/bin/env python3
"""
GAVD Video Downloader (CLI)
- Reads GAVD CSVs, dedupes by `id`, downloads YouTube videos with yt-dlp
- Resumable via processed.txt + skip-existing checks
- Keeps output as a single .mp4 (no stray .m4a/.webm/.info.json files)

Usage:
  python tools/GAVD_video_downloader.py \
    --data-dir GAVD/data \
    --out-dir GAVD-videos \
    --csv GAVD_Clinical_Annotations_1.csv \
    --csv GAVD_Clinical_Annotations_2.csv \
    --csv GAVD_Clinical_Annotations_3.csv \
    --csv GAVD_Clinical_Annotations_4.csv \
    --csv GAVD_Clinical_Annotations_5.csv \
    --processed data_outputs/processed_downloads.txt \
    --failed data_outputs/failed_downloads.txt \
    --skip-existing \
    --sleep 1 \
    --max-videos 0

Notes:
- If your network blocks SSH/22 it doesn't matter here (HTTPS requests only).
- If some videos are age-restricted/region-locked, pass --cookies path/to/cookies.txt
"""
import argparse, os, sys, time, csv
from pathlib import Path
from collections import defaultdict

import pandas as pd
import yt_dlp


def parse_args():
    ap = argparse.ArgumentParser(description="Batch download GAVD videos with yt-dlp.")
    ap.add_argument("--data-dir", default="GAVD/data", help="Directory containing GAVD CSVs")
    ap.add_argument("--csv", action="append", default=[], help="One or more CSV file names inside --data-dir")
    ap.add_argument("--out-dir", default="GAVD-videos", help="Directory to save mp4 files")
    ap.add_argument("--processed", default="data_outputs/processed_downloads.txt", help="Checkpoint file")
    ap.add_argument("--failed", default="data_outputs/failed_downloads.txt", help="Failures log file")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if mp4 already exists")
    ap.add_argument("--cookies", default=None, help="Path to cookies.txt (optional)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between downloads")
    ap.add_argument("--max-videos", type=int, default=0, help="Limit number of videos to download (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true", help="List what would be downloaded, then exit")
    return ap.parse_args()


def load_manifest(data_dir: Path, csv_names):
    if not csv_names:
        # default to 1..5 like your notebook
        csv_names = [f"GAVD_Clinical_Annotations_{i}.csv" for i in range(1, 6)]

    all_videos = {}              # id -> url
    video_info = defaultdict(dict)

    for name in csv_names:
        fp = data_dir / name
        print(f"📁 Loading {fp} ...")
        if not fp.exists():
            print(f"   ❌ Missing file: {fp}")
            continue
        df = pd.read_csv(fp, low_memory=False)
        if "id" not in df.columns or "url" not in df.columns:
            print(f"   ❌ {name} lacks required columns 'id' and 'url' (has: {list(df.columns)})")
            continue
        valid = df[df["url"].notna() & df["id"].notna()]
        for _, row in valid.iterrows():
            vid = str(row["id"]).strip()
            url = str(row["url"]).strip()
            if vid and url and vid not in all_videos:
                all_videos[vid] = url
                video_info[vid] = {
                    "gait_pat": row.get("gait_pat", "unknown"),
                    "dataset": row.get("dataset", "unknown"),
                    "source_file": name,
                }

        print(f"   ✓ Found {valid['id'].nunique()} unique videos in {name}")

    print(f"\n📊 SUMMARY: {len(all_videos)} total unique videos")
    return all_videos, video_info


def read_set(path: Path):
    s = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s.add(line.strip())
    return s


def append_line(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def build_ydl_opts(out_dir: Path, cookies=None):
    opts = {
        # Prefer MP4-only outputs; fall back sanely
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
            "best[ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "ignoreerrors": True,      # continue on individual errors
        "no_warnings": False,
        "writethumbnail": False,
        "writeinfojson": False,    # no .info.json
        "keepvideo": False,        # no leftover separate streams
        "continuedl": True,        # resume partial downloads
        "retries": 10,
        "fragment_retries": 10,
        # Slight politeness; you can tune further if throttled
        "ratelimit": 0,
        "concurrent_fragment_downloads": 1,
    }
    if cookies:
        opts["cookiefile"] = cookies
    return opts


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = Path(args.processed)
    failed_path = Path(args.failed)

    all_videos, video_info = load_manifest(data_dir, args.csv)

    processed = read_set(processed_path)
    failed = read_set(failed_path)

    # Filter out already processed or already present (if skip-existing)
    ids = list(all_videos.keys())
    to_download = []
    for vid in ids:
        mp4 = out_dir / f"{vid}.mp4"
        if vid in processed:
            continue
        if args.skip_existing and mp4.exists() and mp4.stat().st_size > 0:
            # Backfill checkpoint if file exists
            append_line(processed_path, vid)
            continue
        to_download.append(vid)

    if args.max_videos and args.max_videos > 0:
        to_download = to_download[: args.max_videos]

    print(f"\n🔽 Ready to download: {len(to_download)} videos")
    if args.dry_run:
        for i, vid in enumerate(to_download[:20], 1):
            print(f"  {i:>4}. {vid} → {all_videos[vid]}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download)-20} more")
        return 0

    ydl_opts = build_ydl_opts(out_dir, cookies=args.cookies)

    ok, fail = 0, 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, vid in enumerate(to_download, 1):
            url = all_videos[vid]
            gait_type = video_info.get(vid, {}).get("gait_pat", "unknown")
            print(f"\n[{i}/{len(to_download)}] {vid} ({gait_type})")
            print(f"URL: {url}")
            try:
                ydl.download([url])
                # verify file exists
                mp4 = out_dir / f"{vid}.mp4"
                if mp4.exists() and mp4.stat().st_size > 0:
                    append_line(processed_path, vid)
                    ok += 1
                    print(f"✅ Saved: {mp4}")
                else:
                    raise RuntimeError("Download reported OK but .mp4 not found/empty")
            except Exception as e:
                fail += 1
                append_line(failed_path, f"{vid}\t{e}")
                print(f"❌ Failed {vid}: {e}", file=sys.stderr)
            time.sleep(max(0.0, args.sleep))

    print("\n🎉 DONE")
    print(f"✅ Success: {ok}")
    print(f"❌ Failed : {fail}")
    print(f"📁 Output : {out_dir}")
    print(f"🧾 Log    : {processed_path} (processed), {failed_path} (failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
