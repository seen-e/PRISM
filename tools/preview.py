"""Preview the static site, with byte-range support for video chapter seeking.

Run: python tools/preview.py [port]
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import re
import sys


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.range_remaining = None
        path = self.translate_path(self.path)
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last or size))
        end = min(int(last), size - 1) if first and last else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        source = open(path, "rb")
        source.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.range_remaining = end - start + 1
        return source

    def copyfile(self, source, outputfile):
        try:
            if self.range_remaining is None:
                return super().copyfile(source, outputfile)
            while self.range_remaining:
                data = source.read(min(1024 * 1024, self.range_remaining))
                if not data:
                    break
                outputfile.write(data)
                self.range_remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"PRISM preview: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
