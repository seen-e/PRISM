"""Compose complete PPS originals at 1x without resizing their 960x624 panels.

Requires Pillow and FFmpeg. Shorter panels hold their final frame.
The PPS pixels and robot footage are always processed together.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--font', default='C:/Windows/Fonts/arial.ttf')
    args = parser.parse_args()
    out = ROOT / 'assets/video/source'
    out.mkdir(exist_ok=True, parents=True)
    panels = [('PPS插线.mp4','pps-drive-insertion','Drive Connector Insertion',0,56,556),
              ('PPS对插.mp4','pps-bimanual-mating','Bimanual Connector Mating',960,56,517),
              ('PPS拧螺丝.mp4','pps-screw-tightening','Screw Tightening',480,736,675)]
    canvas = Image.new('RGB', (1920,1360), '#10161e')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(args.font, 29)
    for name, slug, title, x, y, count in panels:
        draw.text((x+480, y-34), title, font=font, fill='#f4f7fc', anchor='mm')
        shutil.copy2(Path(args.source_dir)/name, out/(slug+'.mp4'))
    background = out/'pps-background.png'
    canvas.save(background)
    cmd = [args.ffmpeg,'-v','error','-y','-loop','1','-framerate','15','-i',str(background)]
    for name,*_ in panels:
        cmd += ['-i', str(Path(args.source_dir)/name)]
    graph = ''
    for i,(_,_,_,x,y,count) in enumerate(panels,1):
        graph += f'[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=45[p{i}];'
    graph += '[0:v][p1]overlay=0:56:shortest=1[b1];[b1][p2]overlay=960:56:shortest=1[b2];[b2][p3]overlay=480:736:shortest=1[v]'
    cmd += ['-filter_complex_threads','2','-filter_complex',graph,'-map','[v]','-frames:v','675',
            '-an','-c:v','libx264','-preset','fast','-crf','14','-threads','4','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(out/'pps-teleoperation.mp4')]
    subprocess.run(cmd,check=True)
    subprocess.run([args.ffmpeg,'-v','error','-y','-ss','16','-i',str(out/'pps-teleoperation.mp4'),
                    '-frames:v','1','-quality','94',str(ROOT/'assets/images/clips/pps-teleoperation-source.webp')],check=True)
    print('PPS READY: 45 seconds, 15 fps, 1920x1360, native 960x624 panels at 1x.',flush=True)
