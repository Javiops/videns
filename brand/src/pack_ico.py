"""Pack PNG renders into .ico files (PNG-in-ICO: Windows Vista+ and every browser read it)."""
import struct
from pathlib import Path

OUT = Path(__file__).parent / "out"


def ico(path, sizes):
    pngs = [(n, (OUT / f"v-mark-{n}.png").read_bytes()) for n in sizes]
    head = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for n, data in pngs:
        side = 0 if n >= 256 else n                     # 0 means 256 in the directory
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    path.write_bytes(head + entries + blobs)
    print(path.name, sizes, path.stat().st_size, "bytes")


ico(OUT / "favicon.ico", [16, 32, 48])                            # the browser tab
ico(OUT / "videns-app.ico", [16, 24, 32, 48, 64, 128, 256])       # the Windows app, all shell sizes
