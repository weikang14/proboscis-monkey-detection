"""Serve web/ locally with HTTP Range support, so <video> can seek.

`python -m http.server` ignores Range headers, which leaves videos unseekable. Usage:

    python tools/serve.py [port]      # then open http://127.0.0.1:8765
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    """Adds single byte-range responses (206) on top of the stdlib static handler."""

    def send_head(self):
        self._remaining = None
        m = _RANGE.fullmatch(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not m or m.groups() == ("", "") or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = m.groups()
        if first == "":                       # suffix range: last N bytes
            start, end = max(size - int(last), 0), size - 1
        else:
            start, end = int(first), min(int(last) if last else size - 1, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        while self._remaining > 0:
            chunk = source.read(min(64 * 1024, self._remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self._remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    # On Windows SO_REUSEADDR lets a second server bind a busy port and silently lose its traffic.
    ThreadingHTTPServer.allow_reuse_address = sys.platform != "win32"
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=str(WEB)))
    print(f"Serving {WEB} at http://127.0.0.1:{port}")
    httpd.serve_forever()
