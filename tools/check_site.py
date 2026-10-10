"""Validate local assets and anchor references without third-party packages."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            if identifier in self.ids:
                self.errors.append(f"Duplicate ID: {identifier}")
            self.ids.add(identifier)
        for name in ("src", "href", "poster", "data-image"):
            if attrs.get(name):
                self.references.append(attrs[name])
        for name in ("data-play", "data-expand"):
            if attrs.get(name):
                self.references.append("#" + attrs[name])


page = Page()
page.feed((ROOT / "index.html").read_text(encoding="utf-8"))
for stylesheet in ROOT.glob('*.css'):
    page.references.extend(re.findall(r"url\(['\"]?([^)'\"]+)", stylesheet.read_text(encoding="utf-8")))
assets = set()
for value in page.references:
    url = urlsplit(value)
    if url.scheme or url.netloc:
        continue
    if url.path:
        target = ROOT / unquote(url.path)
        assets.add(target)
        if not target.is_file():
            page.errors.append(f"Missing asset: {value}")
    elif url.fragment and url.fragment not in page.ids:
        page.errors.append(f"Missing anchor: {value}")

for asset in assets:
    if asset.exists() and asset.stat().st_size >= 100 * 1024 * 1024:
        page.errors.append(f"Over GitHub per-file limit: {asset.name}")

if page.errors:
    raise SystemExit("\n".join(page.errors))
print(f"PASS: {len(assets)} unique local assets; {len(page.ids)} IDs; all links valid.")
print("Demo size:", (ROOT / "assets/video/prism-demo.mp4").stat().st_size, "bytes")
