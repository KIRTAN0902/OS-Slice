"""
OS-Slice Video Demonstration Recorder
Records a seamless, high-definition (1080p) walkthrough of the OS-Slice UI
spanning the Claude Code CLI showcase and the interactive Streamlit Research Console.
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
RAW_DIR = WORKSPACE / "recorded_demo_raw"
RAW_DIR.mkdir(exist_ok=True)

FINAL_MP4 = WORKSPACE / "OS_Slice_UI_Demonstration.mp4"
THUMBNAIL_PNG = WORKSPACE / "OS_Slice_UI_Thumbnail.png"
ANIM_GIF = WORKSPACE / "OS_Slice_UI_Demonstration.gif"

def smooth_scroll(page, start_y, end_y, steps=25, delay=0.03):
    """Executes smooth continuous scrolling between coordinates."""
    delta = (end_y - start_y) / steps
    for i in range(steps):
        curr = start_y + delta * (i + 1)
        page.evaluate(f"window.scrollTo(0, {curr});")
        time.sleep(delay)

def record():
    print("=================================================================")
    print(" Starting OS-Slice UI Video Demonstration Recording (1080p)")
    print("=================================================================")

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

        # -------------------------------------------------------------
        # SCENE 1: Project Overview & Claude Code CLI Showcase
        # -------------------------------------------------------------
        print("\n[Scene 1] Recording OS-Slice Overview & CLI Showcase...")
        terminal_url = f"file:///{str(WORKSPACE / 'terminal_showcase.html').replace(os.sep, '/')}"
        page.goto(terminal_url)
        page.wait_for_timeout(1500)
        
        # Smooth scroll down to highlight CLI scorecard
        smooth_scroll(page, 0, 300, steps=20, delay=0.04)
        time.sleep(4.0)
        smooth_scroll(page, 300, 0, steps=15, delay=0.04)
        time.sleep(2.0)

        # -------------------------------------------------------------
        # SCENE 2: Transition to Streamlit Interactive Research Console
        # -------------------------------------------------------------
        print("[Scene 2] Loading Streamlit Research Console on http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)

        print("[*] Waiting for model artifacts & initial evaluation to complete...")
        try:
            page.wait_for_selector('[data-testid="stSpinner"]', state="detached", timeout=60000)
        except Exception:
            pass

        time.sleep(2.0)
        # Capture thumbnail here at main dashboard view
        page.screenshot(path=str(THUMBNAIL_PNG))
        print(f"[✓] Saved high-resolution thumbnail to {THUMBNAIL_PNG.name}")

        # Smooth scroll through top banner and specifications
        smooth_scroll(page, 0, 250, steps=20, delay=0.03)
        time.sleep(2.5)

        # -------------------------------------------------------------
        # SCENE 3: Tab 1 — Live Side-by-Side Generation Console
        # -------------------------------------------------------------
        print("[Scene 3] Interacting with Tab 1: Live Side-by-Side Generation...")
        tab1 = page.get_by_text("Side-by-Side Generation").first
        tab1.click()
        time.sleep(1.5)

        # Scroll down to inspect Base Model vs OS-Slice Micro-Model outputs
        smooth_scroll(page, 250, 650, steps=30, delay=0.03)
        time.sleep(4.5)  # Let mentor examine the strict JSON blocks and metrics

        # Scroll further down to inspect schema adherence diff
        smooth_scroll(page, 650, 950, steps=25, delay=0.03)
        time.sleep(3.5)

        # -------------------------------------------------------------
        # SCENE 4: Tab 2 — Behavioral Circuit Saliency
        # -------------------------------------------------------------
        print("[Scene 4] Navigating to Tab 2: Behavioral Circuit Saliency...")
        smooth_scroll(page, 950, 0, steps=25, delay=0.02)
        time.sleep(1.0)
        
        tab2 = page.get_by_text("Behavioral Circuit Saliency").first
        tab2.click()
        time.sleep(2.0)

        # Scroll through Fisher Saliency heatmaps & layer distributions
        smooth_scroll(page, 0, 450, steps=25, delay=0.03)
        time.sleep(4.0)
        smooth_scroll(page, 450, 850, steps=25, delay=0.03)
        time.sleep(3.5)

        # -------------------------------------------------------------
        # SCENE 5: Tab 3 — Empirical Benchmarks Scorecard
        # -------------------------------------------------------------
        print("[Scene 5] Navigating to Tab 3: Empirical Benchmarks Scorecard...")
        smooth_scroll(page, 850, 0, steps=25, delay=0.02)
        time.sleep(1.0)

        tab3 = page.get_by_text("Empirical Benchmarks").first
        tab3.click()
        time.sleep(2.0)

        # Inspect 100% strict parseability, throughput, memory savings
        smooth_scroll(page, 0, 400, steps=20, delay=0.03)
        time.sleep(4.0)
        smooth_scroll(page, 400, 750, steps=20, delay=0.03)
        time.sleep(3.5)

        # -------------------------------------------------------------
        # SCENE 6: Tab 4 — Foundational Model Catalog (360M to 120B)
        # -------------------------------------------------------------
        print("[Scene 6] Navigating to Tab 4: Foundational Model Catalog (360M to 120B)...")
        smooth_scroll(page, 750, 0, steps=25, delay=0.02)
        time.sleep(1.0)

        tab4 = page.get_by_text("Model Catalog (SLMs to 120B)").first
        tab4.click()
        time.sleep(2.0)

        # Scroll through tiered models and VRAM reduction calculator
        smooth_scroll(page, 0, 500, steps=25, delay=0.03)
        time.sleep(4.0)
        smooth_scroll(page, 500, 950, steps=25, delay=0.03)
        time.sleep(4.0)

        # -------------------------------------------------------------
        # SCENE 7: Return & Final Conclusion
        # -------------------------------------------------------------
        print("[Scene 7] Finalizing demonstration recording...")
        smooth_scroll(page, 950, 0, steps=30, delay=0.02)
        time.sleep(2.5)

        context.close()
        browser.close()
        print("[✓] Raw WebM capture completed and flushed to disk.")

    # -----------------------------------------------------------------
    # SCENE 8: FFmpeg Post-Processing to MP4 and GIF
    # -----------------------------------------------------------------
    webm_files = list(RAW_DIR.glob("*.webm"))
    if not webm_files:
        print("[!] Error: No WebM recording found in raw directory.")
        return

    # Select latest webm file
    latest_webm = max(webm_files, key=lambda f: f.stat().st_mtime)
    print(f"\n[*] Transcoding {latest_webm.name} to MP4 using FFmpeg (H.264, 1080p, 30fps)...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(latest_webm),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        str(FINAL_MP4)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    mp4_size_mb = FINAL_MP4.stat().st_size / (1024 * 1024)
    print(f"[✓] Successfully produced MP4 video: {FINAL_MP4} ({mp4_size_mb:.2f} MB)")

    # Produce compressed preview GIF (first 15s) for instant previews/embedding
    print("[*] Generating animated GIF preview for documentation & README...")
    gif_cmd = [
        "ffmpeg", "-y",
        "-ss", "00:00:08",
        "-t", "12",
        "-i", str(FINAL_MP4),
        "-vf", "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0",
        str(ANIM_GIF)
    ]
    try:
        subprocess.run(gif_cmd, check=True)
        gif_size_mb = ANIM_GIF.stat().st_size / (1024 * 1024)
        print(f"[✓] Generated animated GIF preview: {ANIM_GIF} ({gif_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] Warning: GIF generation skipped ({e})")

    print("\n=================================================================")
    print(" OS-Slice Video Recording & Rendering Complete!")
    print(f" • MP4 Video:  {FINAL_MP4}")
    print(f" • Thumbnail:  {THUMBNAIL_PNG}")
    print(f" • GIF Preview: {ANIM_GIF}")
    print("=================================================================")

if __name__ == "__main__":
    record()
