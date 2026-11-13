"""Download the part of comma2k19 used by the worked application of Chapter 4 (standard library only).

comma2k19 (Schafer et al., 2018; MIT licence) is distributed as ten zip archives of about 9 GB each, which also
contain the videos. This script reads the central directory of each archive with HTTP Range requests and fetches
only the processed logs that the example needs (u-blox GNSS fixes and raw measurements, phone IMU, CAN speed and
wheel speeds, reference poses) for the routes listed in ROUTES below: about 80 MB in total.

usage:  python download_comma2k19.py [--out DIR] [--max-seg N]
        (default DIR: ../../data_cache/comma2k19 relative to this script; N: 12 consecutive minutes per route)

The archives are hosted by Hugging Face (dataset commaai/comma2k19). The data are only read with numpy
(allow_pickle=False) by the analysis script.
"""
import argparse
import io
import os
import re
import sys
import time
import urllib.request
import zipfile

BASE = "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{}.zip"

# (chunk, route, role): RAV4 = dongle b0c9d2329ad1606b (chunks 1-2), Civic = 99c94dc769b5d96e (chunks 3-10)
ROUTES = [
    (3, "99c94dc769b5d96e|2018-05-18--16-00-42", "train"),
    (5, "99c94dc769b5d96e|2018-07-05--20-49-41", "train"),
    (7, "99c94dc769b5d96e|2018-08-01--14-45-10", "train"),
    (9, "99c94dc769b5d96e|2018-10-12--20-13-54", "train"),
    (3, "99c94dc769b5d96e|2018-05-03--08-13-12", "test"),
    (5, "99c94dc769b5d96e|2018-07-03--14-38-26", "test"),
    (9, "99c94dc769b5d96e|2018-10-15--14-39-02", "test"),
    (1, "b0c9d2329ad1606b|2018-08-17--12-07-08", "rav4"),
    (1, "b0c9d2329ad1606b|2018-08-14--10-32-01", "rav4"),
    (2, "b0c9d2329ad1606b|2018-10-09--15-48-37", "rav4"),
    (2, "b0c9d2329ad1606b|2018-11-11--13-08-44", "rav4"),
]
MEMBERS = re.compile(r"/(processed_log/(GNSS/(live_gnss_ublox|raw_gnss_ublox)|IMU/(gyro|accelerometer)|"
                     r"CAN/(speed|wheel_speed))|global_pose)/[^/]+$")


class RangedHTTPFile(io.RawIOBase):
    """Read-only, seekable view of a remote file; bytes are fetched with HTTP Range requests."""

    def __init__(self, url, block=1 << 16):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos, self.block, self.cache, self.bytes = 0, block, [], 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def fetch(self, start, stop):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception:  # transient network error: retry
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        self.bytes += len(data)
        self.cache.append((start, data))
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        stop = min(self.pos + n, self.size)
        if stop <= self.pos:
            return b""
        for s, d in self.cache:
            if s <= self.pos and stop <= s + len(d):
                out = d[self.pos - s:stop - s]
                self.pos = stop
                return out
        d = self.fetch(self.pos, min(self.size, max(stop, self.pos + self.block)))
        out = d[:stop - self.pos]
        self.pos = stop
        return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "data_cache", "comma2k19"))
    ap.add_argument("--max-seg", type=int, default=12)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    for chunk in sorted({c for c, _, _ in ROUTES}):
        routes = {r: role for c, r, role in ROUTES if c == chunk}
        f = RangedHTTPFile(BASE.format(chunk))
        zf = zipfile.ZipFile(f)
        cand = [i for i in zf.infolist()
                if len(i.filename.split("/")) >= 4 and i.filename.split("/")[1] in routes and MEMBERS.search(i.filename)]
        # the archives hold a subset of the segments of each route: keep the first max_seg of them
        keep = {}
        for i in cand:
            route, seg = i.filename.split("/")[1:3]
            keep.setdefault(route, set()).add(int(seg))
        keep = {r: set(sorted(s)[:args.max_seg]) for r, s in keep.items()}
        sel = [i for i in cand if int(i.filename.split("/")[2]) in keep[i.filename.split("/")[1]]]
        # fetch the selected members in a few large contiguous ranges
        spans = sorted((i.header_offset, i.header_offset + 30 + len(i.orig_filename.encode()) + len(i.extra)
                        + 1024 + i.compress_size) for i in sel)
        merged = []
        for s, e in spans:
            if merged and s - merged[-1][1] < (512 << 10):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            f.fetch(s, min(e, f.size))
        for info in sel:
            parts = info.filename.split("/")
            role = routes[parts[1]]
            dest = os.path.join(out, role, parts[1].replace("|", "_"), *parts[2:])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
        print(f"Chunk_{chunk}: {len(sel)} files from {len(routes)} route(s), {f.bytes / 1e6:.1f} MB transferred")
    print("data in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
