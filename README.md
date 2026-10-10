# PRISM project website

Project website for **PRISM: Precision Forcing for Generalist Vision-Language-Action Policies**.

Live GitHub Pages URL: **https://detimage123.github.io/PRISM/**

The paper has not been uploaded to arXiv. The page labels it as a **research manuscript** and uses an `@unpublished` BibTeX entry without an arXiv identifier.

## Files

- `index.html`: paper title, authors, affiliations, method, results, and citation.
- `styles.css`: responsive desktop/mobile design.
- `research.css`: SEEN·E / CSDW research-page theme, with locally hosted Prompt fonts, neutral backgrounds, red accents, and the original PRISM blue wordmark and logo.
- `app.js`: independent video playback, enlarged video/image views, citation copying.
- `assets/paper/PRISM.pdf`: 21-page PDF compiled from the supplied `PRISM_arXiv_preprint_20261008.zip` without changing its LaTeX source.
- `assets/paper/prism.bib`: citation for the current unpublished manuscript.
- `assets/video/prism-demo.mp4`: approved 180-second, 1080p/30fps demonstration; original exported file copied without re-encoding (76,060,560 bytes).
- `assets/video/source/*.mp4`: 1× videos rebuilt directly from the supplied original recordings at their native resolutions and frame rates; no 3-minute-film slices are used by the page.
- `assets/source-clips.json`: original filenames, frame ranges, native dimensions, and 1× durations.
- `assets/video/source/pps-teleoperation.mp4`: archived 45-second PPS composition, no longer referenced by the homepage.
- `assets/video/source/pps-{drive-insertion,bimanual-mating,screw-tightening}.mp4`: untouched PPS original files, displayed in three independent players with full trajectories and synchronized score overlays.
- `assets/clips.json` and the old top-level MP4s: archived previous website media; no longer referenced by the homepage.
- `assets/images/logo.webp`: the supplied PRISM logo, used in the title, navigation, footer, and browser favicon.
- `assets/fonts/`: locally hosted Manrope fonts and their SIL Open Font License.
- `assets/chapters.vtt`: chapters matching the final demonstration.
- `tools/preview.py`: local preview server with video byte-range support.
- `tools/check_site.py`: local file and anchor validation.
- `tools/build_source_videos.py` and `tools/source_videos.json`: reproducible native-resolution H.264 conversion, without speed changes or resampling.
- `tools/build_pps_source.py`: full-resolution 1× PPS composition.
- `tools/split_videos.py`: archived previous 3-minute-film splitting workflow.

There are no external JavaScript, CSS, font, analytics, or build dependencies. Videos load when requested rather than downloading during initial page load. All site paths are relative and work under `/PRISM/`.

Main task videos and PPS videos each form a three-column row on desktop and stack vertically on mobile. PPS enlarged playback retains the Human VR Teleoperation label.

## Local preview

With Python 3 installed, run from this directory:

```sh
python tools/preview.py
```

Open http://127.0.0.1:8765/. Use this server for testing video seeking; a server without HTTP byte-range support may make seeking appear broken.

## GitHub Pages

Upload these files to the `main` branch of `detimage123/PRISM`, keeping `index.html` at the repository root and including `.nojekyll`.

In **Settings → Pages**, select **Deploy from a branch**, then **main / (root)**. GitHub serves the site at https://detimage123.github.io/PRISM/ after deployment finishes.

Current media uses browser-compatible H.264 with fast-start MP4 layout. A 1× label means the timing of the supplied original file; no additional speed-up or slow-down is applied. Audio is omitted, and the previously identified network label is masked in the local-alignment video. Full recordings are retained; the 84-second perturbation recording is split at 30 seconds into two consecutive videos without omitting frames. The original three-minute submission film remains archived, unchanged.

## Editing

Edit HTML content directly, replace figure/media assets using the same names, and adjust colors/spacing in `styles.css`. The citation appears both in `index.html` and `assets/paper/prism.bib`; update both together.

Paper results follow the provided manuscript, including the trial count and the additional demonstrations used for DAgger. The disturbance clips are qualitative examples. PPS is described as a trajectory-derived proxy, not a physical tolerance measurement. There is no arXiv identifier, acceptance badge, or code-release claim.

The current visual template follows the company's [CSDW project page](https://seen-e.github.io/CSDW/): Prompt typography, black/gray text, pale gray sections, restrained red accents, fine dividers, and squared buttons. PRISM's logo artwork and blue title gradient remain unchanged. Main tasks and the three independent PPS recordings retain their existing desktop rows and responsive mobile layouts.

The intended destination is `https://seen-e.github.io/PRISM/`. Repository transfer is pending organization permission; on 2026-10-10 GitHub rejected transfer because detimage123 cannot create public repositories on seen-e. Keep canonical, repository, citation, and release URLs at their working locations until transfer succeeds. After transfer, update those URLs to seen-e and verify Pages plus release-video playback.

## Checks

```sh
python tools/check_site.py
node --check app.js
```

Check desktop and mobile layouts, each video, enlarged playback, figure enlargement, citation copying, and the paper download after replacing any assets.

Native-resolution MP4s above GitHub's per-file repository limit are hosted in the same repository's `website-media-20261008` release, with `video/mp4` content type. The homepage links those assets directly. Local copies are preserved under `assets/video/source/` and excluded from Git; `assets/source-clips.json` records their local and hosted paths.
