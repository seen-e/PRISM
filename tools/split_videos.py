"""Split the approved 3-minute film at exact frame boundaries.

python tools/split_videos.py --ffmpeg /path/to/ffmpeg
No action frames or embedded speed/PPS labels are removed inside a clip.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
CLIPS = [
    ("drive-insertion", "Drive Connector Insertion", "Main tasks", 105, 593),
    ("connector-close-up", "Connector Close-up", "Alignment", 593, 787),
    ("local-alignment", "Local Alignment under Challenging Poses", "Alignment", 787, 1324),
    ("grasp-perturbation", "Grasp Perturbation", "Perturbations", 1324, 1616),
    ("insertion-perturbation", "Insertion Perturbation", "Perturbations", 1616, 1926),
    ("target-displacement", "Target Displacement", "Perturbations", 1926, 2166),
    ("visual-distraction", "Visual Distraction", "Perturbations", 2166, 2476),
    ("lighting-changes", "Lighting Changes", "Perturbations", 2476, 2959),
    ("bimanual-mating", "Bimanual Connector Mating", "Main tasks", 2959, 4136),
    ("screw-tightening", "Screw Tightening", "Main tasks", 4136, 4657),
    ("pps-teleoperation", "Proprioceptive Precision Scoring", "Human VR Teleoperation", 4657, 5332),
]


def render(clip, ffmpeg):
    slug, title, group, start, end = clip
    source = ROOT / "assets/video/prism-demo.mp4"
    dest = ROOT / f"assets/video/{slug}.mp4"
    if slug != "pps-teleoperation" or not dest.exists():
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start / 30:.10f}", "-i", str(source),
            "-frames:v", str(end - start), "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-threads", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest),
        ], check=True)
    poster = ROOT / f"assets/images/clips/{slug}.webp"
    poster.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss",
        f"{min((end - start) / 30 * .55, 8):.4f}", "-i", str(dest),
        "-frames:v", "1", "-vf", "scale=960:-2", "-quality", "90", str(poster),
    ], check=True)
    print(f"Ready: {slug} ({end-start} frames, {(end-start)/30:.3f} seconds)", flush=True)
    return {"id": slug, "title": title, "group": group,
            "source": "prism-demo.mp4", "start_frame": start, "end_frame_exclusive": end,
            "fps": 30, "duration": (end-start)/30, "frames": end-start,
            "video": f"assets/video/{slug}.mp4", "poster": f"assets/images/clips/{slug}.webp"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        clips = list(pool.map(lambda c: render(c, args.ffmpeg), CLIPS))
    (ROOT / "assets/clips.json").write_text(json.dumps(clips, indent=2) + "\n", encoding="utf-8")
    print("All clips cover frames 105–5331 consecutively. Only opening/closing title cards are excluded.")
