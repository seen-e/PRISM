# Independent experiment videos

Source: the approved `assets/video/prism-demo.mp4`, 1920×1080 at 30 fps. Boundaries below refer to this full film, with the end exclusive. Each file starts at local time zero.

| Clip | Full-film interval | Frames | Duration | Embedded speed |
| --- | --- | ---: | ---: | --- |
| Drive Connector Insertion | 00:03.500–00:19.767 | 488 | 16.267 s | 2× |
| Connector Close-up | 00:19.767–00:26.233 | 194 | 6.467 s | 2× |
| Local Alignment | 00:26.233–00:44.133 | 537 | 17.900 s | 2× |
| Grasp Perturbation | 00:44.133–00:53.867 | 292 | 9.733 s | 2× |
| Insertion Perturbation | 00:53.867–01:04.200 | 310 | 10.333 s | 2× / 1× |
| Target Displacement | 01:04.200–01:12.200 | 240 | 8.000 s | 2× |
| Visual Distraction | 01:12.200–01:22.533 | 310 | 10.333 s | 2× |
| Lighting Changes | 01:22.533–01:38.633 | 483 | 16.100 s | 2× |
| Bimanual Connector Mating | 01:38.633–02:17.867 | 1177 | 39.233 s | 1× / 2× |
| Screw Tightening | 02:17.867–02:35.233 | 521 | 17.367 s | 1× / 2× |
| Proprioceptive Precision Scoring | 02:35.233–02:57.733 | 675 | 22.500 s | 2× |

The first 10 clips are PRISM inference experiments. PPS uses human VR teleoperation, retaining all three complete synchronized trajectories in the original panel.

These 5,227 consecutive frames cover every experimental frame in the approved film. Only its opening and closing title cards are excluded from the individual clips. No playback speed, robot motion, score overlay, or temporal relationship was changed. The complete 180-second film remains available as a separate download.

`assets/clips.json` records the exact integer frame boundaries and filenames. `tools/split_videos.py` reproduces the split using FFmpeg.
