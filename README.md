# PRISM project website

Project website for **PRISM: Precision Forcing for Generalist Vision-Language-Action Policies**.

Live GitHub Pages URL: **https://detimage123.github.io/PRISM/**

The paper has not been uploaded to arXiv. The page labels it as a **research manuscript** and uses an `@unpublished` BibTeX entry without an arXiv identifier.

## Files

- `index.html`: paper title, authors, affiliations, method, results, and citation.
- `styles.css`: responsive desktop/mobile design.
- `app.js`: independent video playback, enlarged video/image views, citation copying.
- `assets/paper/PRISM.pdf`: 21-page PDF compiled from the supplied `PRISM_arXiv_preprint_20261008.zip` without changing its LaTeX source.
- `assets/paper/prism.bib`: citation for the current unpublished manuscript.
- `assets/video/prism-demo.mp4`: approved 180-second, 1080p/30fps demonstration; original exported file copied without re-encoding (76,060,560 bytes).
- `assets/video/pps-teleoperation.mp4`: the complete 22.5-second PPS panel from the approved demonstration, labeled Human VR Teleoperation, with original overlays and 2× timing preserved.
- `assets/video/*.mp4`: 10 additional standalone experiment clips, grouped into main tasks, alignment, and perturbations. Every experimental frame from the approved full film is retained across the 11 clips; only its opening and closing title cards are excluded.
- `assets/clips.json`: precise frame boundaries, durations, titles, and media paths.
- `assets/images/logo.webp`: the supplied PRISM logo, used in the title, navigation, footer, and browser favicon.
- `assets/fonts/`: locally hosted Manrope fonts and their SIL Open Font License.
- `assets/chapters.vtt`: chapters matching the final demonstration.
- `tools/preview.py`: local preview server with video byte-range support.
- `tools/check_site.py`: local file and anchor validation.
- `tools/split_videos.py`: reproducible frame-accurate video splitting with FFmpeg.

There are no external JavaScript, CSS, font, analytics, or build dependencies. Videos load when requested rather than downloading during initial page load. All site paths are relative and work under `/PRISM/`.

## Local preview

With Python 3 installed, run from this directory:

```sh
python tools/preview.py
```

Open http://127.0.0.1:8765/. Use this server for testing video seeking; a server without HTTP byte-range support may make seeking appear broken.

## GitHub Pages

Upload these files to the `main` branch of `detimage123/PRISM`, keeping `index.html` at the repository root and including `.nojekyll`.

In **Settings → Pages**, select **Deploy from a branch**, then **main / (root)**. GitHub serves the site at https://detimage123.github.io/PRISM/ after deployment finishes.

The MP4 is below GitHub's 100 MiB per-file limit and is stored directly, not through Git LFS. Keep future video replacements below this limit, update chapter timestamps when editing the video, and preserve embedded speed and teleoperation labels.

## Editing

Edit HTML content directly, replace figure/media assets using the same names, and adjust colors/spacing in `styles.css`. The citation appears both in `index.html` and `assets/paper/prism.bib`; update both together.

Paper results follow the provided manuscript, including the trial count and the additional demonstrations used for DAgger. The disturbance clips are qualitative examples. PPS is described as a trajectory-derived proxy, not a physical tolerance measurement. There is no arXiv identifier, acceptance badge, or code-release claim.

The visual design follows the provided logo's navy, ice-blue, cyan, and violet palette. Research-page references informed the structure, without copying their source code or media: [Mobile ALOHA](https://mobile-aloha.github.io/), [OpenVLA](https://openvla.github.io/), [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), and [RDT-1B](https://rdt-robotics.github.io/rdt-robotics/).

## Checks

```sh
python tools/check_site.py
node --check app.js
```

Check desktop and mobile layouts, each video, enlarged playback, figure enlargement, citation copying, and the paper download after replacing any assets.
