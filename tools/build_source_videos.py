"""Build native-resolution, 1x website videos directly from the original recordings.

python tools/build_source_videos.py --source-dir /path/to/originals --ffmpeg ffmpeg
No temporal resampling, speed changes, or spatial downsampling. Audio is omitted.
source_videos.json records the source frame ranges and privacy mask.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

def run(command):
    subprocess.run(command, check=True)

def render(c, args):
    target = ROOT / 'assets/video/source' / (c['id'] + '.mp4')
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = c['end_frame_exclusive'] - c['start_frame']
    duration = frames / c['fps']
    command = [args.ffmpeg, '-v', 'error', '-nostdin', '-y', '-threads', '4',
               '-ss', str(c['start_frame'] / c['fps']), '-i', str(Path(args.source_dir) / c['source'])]
    mapped = '0:v:0'
    if c['privacy_masks']:
        m = c['privacy_masks'][0]
        # Pixelate only the network label, leaving the contact region untouched.
        graph = (f"[0:v]split[base][roi];[roi]crop={m['width']}:{m['height']}:{m['x']}:{m['y']},"
                 f"scale=12:12:flags=area,scale={m['width']}:{m['height']}:flags=neighbor[mask];"
                 f"[base][mask]overlay={m['x']}:{m['y']}:enable='gte(t,{m['start']})'[v]")
        command += ['-filter_complex_threads', '2', '-filter_complex', graph]
        mapped = '[v]'
    command += ['-map', mapped, '-frames:v', str(frames), '-an', '-map_metadata', '-1',
                '-c:v', 'libx264', '-crf', '17', '-preset', 'fast', '-threads', '6',
                '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-movflags', '+faststart', str(target)]
    run(command)
    poster = ROOT / 'assets/images/clips' / (c['id'] + '-source.webp')
    run([args.ffmpeg, '-v', 'error', '-y', '-ss', str(c['poster_time']), '-i', str(target),
         '-frames:v', '1', '-vf', 'scale=1440:-2', '-quality', '94', str(poster)])
    result = dict(c, frames=frames, duration=duration, video=target.relative_to(ROOT).as_posix(),
                  poster=poster.relative_to(ROOT).as_posix(), bytes=target.stat().st_size)
    print(f"READY {c['id']} {duration:.3f}s {c['width']}x{c['height']} {result['bytes']/1e6:.1f} MB", flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--only', help='Rebuild one clip without replacing the complete manifest')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'tools/source_videos.json').read_text(encoding='utf-8'))
    if args.only:
        render(next(c for c in plan if c['id'] == args.only), args)
        raise SystemExit(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        clips = list(pool.map(lambda c: render(c, args), plan))
    (ROOT / 'assets/source-clips.json').write_text(json.dumps(clips, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
