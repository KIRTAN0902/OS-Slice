"""
OS-Slice CLI Exact Screen Recording Runner
Launches Playwright at 1080p Full HD to record the exact terminal CLI experience.
"""

import sys
import io
import os
import time
import subprocess
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from playwright.sync_api import sync_playwright

WORKSPACE = Path(__file__).parent.resolve()
RAW_DIR = WORKSPACE / "recorded_cli_raw"
RAW_DIR.mkdir(exist_ok=True)

FINAL_MP4 = WORKSPACE / "OS_Slice_CLI_Screen_Recording.mp4"
ANIM_GIF = WORKSPACE / "OS_Slice_CLI_Screen_Recording.gif"
THUMBNAIL_PNG = WORKSPACE / "OS_Slice_CLI_Thumbnail.png"

def run_recording():
    print("=================================================================", flush=True)
    print(" Starting OS-Slice CLI Exact Screen Recording (1080p Full HD)", flush=True)
    print("=================================================================", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--disable-infobars", "--no-sandbox"]
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(RAW_DIR),
            record_video_size={"width": 1920, "height": 1080}
        )

        page = context.new_page()

        file_url = f"file:///{str(WORKSPACE / 'cli_recorder.html').replace(os.sep, '/')}"
        print(f"[*] Loading terminal interface: {file_url}", flush=True)
        page.goto(file_url)

        # Wait for initial render and take thumbnail
        page.wait_for_timeout(3500)
        page.screenshot(path=str(THUMBNAIL_PNG))
        print(f"[✓] Saved high-resolution CLI thumbnail: {THUMBNAIL_PNG.name}", flush=True)

        # Wait until the walkthrough completes
        print("[*] Recording interactive CLI session (Menus, Hub, 120B Catalog, Benchmarks, Playground)...", flush=True)
        page.wait_for_function("window.walkthroughComplete === true", timeout=90000)
        page.wait_for_timeout(2000)

        context.close()
        browser.close()
        print("[✓] Raw WebM capture completed and flushed.", flush=True)

    # Transcode WebM to MP4 using FFmpeg
    webm_files = list(RAW_DIR.glob("*.webm"))
    if not webm_files:
        print("[!] Error: No WebM found.")
        return

    latest_webm = max(webm_files, key=lambda f: f.stat().st_mtime)
    print(f"\n[*] Transcoding {latest_webm.name} to MP4 using FFmpeg (H.264, 1080p, 30fps)...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(latest_webm),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        str(FINAL_MP4)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    size_mb = FINAL_MP4.stat().st_size / (1024 * 1024)
    print(f"[✓] Successfully generated MP4 video: {FINAL_MP4} ({size_mb:.2f} MB)")

    # Generate animated GIF preview
    print("[*] Generating animated GIF preview...")
    gif_cmd = [
        "ffmpeg", "-y",
        "-ss", "00:00:03",
        "-t", "14",
        "-i", str(FINAL_MP4),
        "-vf", "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0",
        str(ANIM_GIF)
    ]
    try:
        subprocess.run(gif_cmd, check=True)
        gif_size = ANIM_GIF.stat().st_size / (1024 * 1024)
        print(f"[✓] Generated animated GIF preview: {ANIM_GIF} ({gif_size:.2f} MB)")
    except Exception as e:
        print(f"[!] Warning: GIF failed: {e}")

    print("\n=================================================================")
    print(" OS-Slice CLI Screen Recording Complete!")
    print(f" • MP4 Video:  {FINAL_MP4} ({size_mb:.2f} MB)")
    print(f" • Thumbnail:  {THUMBNAIL_PNG}")
    print(f" • GIF Preview: {ANIM_GIF}")
    print("=================================================================")

if __name__ == "__main__":
    run_recording()
