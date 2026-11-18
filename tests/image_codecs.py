"""Minimal image writers for decoder tests: PNG (all color types, bit depths,
Adam7), BMP, PPM/PGM and JPEG (baseline or progressive, 4:4:4 or 4:2:0,
restart intervals, optimized Huffman tables). NumPy and the standard library only.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np


# ---------------------------------------------------------------- PNG

def _png_chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _pack_row(samples, depth):
    """samples: 1-D integer array of one row (all channels interleaved)."""
    if depth == 8:
        return samples.astype(np.uint8).tobytes()
    if depth == 16:
        return samples.astype(">u2").tobytes()
    per_byte = 8 // depth
    padded = np.zeros((len(samples) + per_byte - 1) // per_byte * per_byte, dtype=np.uint8)
    padded[:len(samples)] = samples
    out = np.zeros(len(padded) // per_byte, dtype=np.uint8)
    for i in range(per_byte):
        out |= (padded[i::per_byte] << (8 - depth * (i + 1))).astype(np.uint8)
    return out.tobytes()


def _filter_rows(rows, bpp):
    """Filters rows with types cycling 0..4 (exercising every filter)."""
    out, previous = bytearray(), bytes(len(rows[0])) if rows else b""
    for y, row in enumerate(rows):
        kind = y % 5
        row = np.frombuffer(row, dtype=np.uint8).astype(np.int32)
        prev = np.frombuffer(previous, dtype=np.uint8).astype(np.int32)
        left = np.concatenate([np.zeros(bpp, np.int32), row[:-bpp]])
        upleft = np.concatenate([np.zeros(bpp, np.int32), prev[:-bpp]])
        if kind == 0:
            filtered = row
        elif kind == 1:
            filtered = row - left
        elif kind == 2:
            filtered = row - prev
        elif kind == 3:
            filtered = row - (left + prev) // 2
        else:
            pa, pb, pc = np.abs(prev - upleft), np.abs(left - upleft), np.abs(left + prev - 2 * upleft)
            predictor = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
            filtered = row - predictor
        out.append(kind)
        out += (filtered & 0xFF).astype(np.uint8).tobytes()
        previous = row.astype(np.uint8).tobytes()
    return bytes(out)


def write_png(path, samples, color, depth=8, palette=None, trns=None, interlace=False):
    """samples: (H, W, channels) integers in the PNG sample range."""
    samples = np.asarray(samples)
    height, width = samples.shape[:2]
    channels = samples.shape[2] if samples.ndim == 3 else 1
    bpp = max(1, channels * depth // 8)
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] \
        if interlace else [(0, 0, 1, 1)]
    raw = b""
    for x0, y0, dx, dy in passes:
        part = samples[y0::dy, x0::dx]
        if part.size == 0:
            continue
        rows = [_pack_row(part[y].reshape(-1), depth) for y in range(part.shape[0])]
        raw += _filter_rows(rows, bpp)
    body = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlace)))
    if palette is not None:
        body += _png_chunk(b"PLTE", np.asarray(palette, np.uint8).tobytes())
    if trns is not None:
        body += _png_chunk(b"tRNS", bytes(trns))
    data = zlib.compress(raw, 9)
    # Split IDAT to exercise chunk concatenation.
    for i in range(0, len(data), 997):
        body += _png_chunk(b"IDAT", data[i:i + 997])
    body += _png_chunk(b"IEND", b"")
    open(path, "wb").write(body)


def over_white(rgb, alpha):
    rgb, alpha = rgb.astype(np.int64), alpha.astype(np.int64)[..., None]
    return ((rgb * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)


# ---------------------------------------------------------------- BMP, PPM

def write_bmp(path, rgb):
    height, width = rgb.shape[:2]
    stride = (width * 3 + 3) & ~3
    pixels = bytearray()
    for y in range(height - 1, -1, -1):
        row = rgb[y, :, ::-1].astype(np.uint8).tobytes()
        pixels += row + bytes(stride - len(row))
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    open(path, "wb").write(header + info + bytes(pixels))


def write_ppm(path, samples, maximum=255):
    samples = np.asarray(samples)
    color = samples.ndim == 3
    height, width = samples.shape[:2]
    head = f"P{6 if color else 5}\n# test image\n{width} {height}\n{maximum}\n".encode()
    data = samples.astype(">u2" if maximum > 255 else np.uint8).tobytes()
    open(path, "wb").write(head + data)


# ---------------------------------------------------------------- JPEG

ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7,
                   14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46,
                   53, 60, 61, 54, 47, 55, 62, 63])
LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
                 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
                 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
CHROMA = np.array([17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66] + [99] * 4 + [24, 26, 56] + [99] * 5 + [47, 66] + [99] * 38)


def _quant(table, quality):
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip((table * scale + 50) // 100, 1, 255).astype(np.int32)


def _dct_matrix():
    m = np.zeros((8, 8))
    for u in range(8):
        for x in range(8):
            m[u, x] = (np.sqrt(0.5) if u == 0 else 1) * np.cos((2 * x + 1) * u * np.pi / 16) / 2
    return m


def _huffman_table(counts):
    """JPEG Annex K.2: code lengths (<= 16) from symbol frequencies."""
    freq = np.zeros(257, dtype=np.int64)
    for symbol, count in counts.items():
        freq[symbol] = count
    freq[256] = 1  # Reserved so no code is all ones.
    codesize, others = np.zeros(257, dtype=np.int64), np.full(257, -1)
    while True:
        candidates = [i for i in range(257) if freq[i] > 0]
        if len(candidates) < 2:
            break
        v1 = min(candidates, key=lambda i: (freq[i], -i))
        v2 = min((i for i in candidates if i != v1), key=lambda i: (freq[i], -i))
        freq[v1] += freq[v2]
        freq[v2] = 0
        for v in (v1, v2):
            node = v
            codesize[node] += 1
            while others[node] >= 0:
                node = others[node]
                codesize[node] += 1
            if v == v1:
                last = node
        others[last] = v2
    bits = np.zeros(33, dtype=np.int64)
    for i in range(257):
        if codesize[i]:
            bits[codesize[i]] += 1
    for i in range(32, 16, -1):
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0:
                j -= 1
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
    i = 16
    while bits[i] == 0:
        i -= 1
    bits[i] -= 1  # Remove the reserved symbol.
    symbols = [s for size in range(1, 33) for s in range(256) if codesize[s] == size]
    lengths = [int(b) for b in bits[1:17]]
    codes, code, k = {}, 0, 0
    for size in range(1, 17):
        for _ in range(lengths[size - 1]):
            codes[symbols[k]] = (code, size)
            code += 1
            k += 1
        code <<= 1
    return lengths, symbols[:sum(lengths)], codes


class _Writer:
    def __init__(self):
        self.out, self.buffer, self.bits = bytearray(), 0, 0

    def put(self, value, n):
        for i in range(n - 1, -1, -1):
            self.buffer = (self.buffer << 1) | ((value >> i) & 1)
            self.bits += 1
            if self.bits == 8:
                self.out.append(self.buffer)
                if self.buffer == 0xFF:
                    self.out.append(0)
                self.buffer, self.bits = 0, 0

    def flush(self):
        if self.bits:
            self.put((1 << (8 - self.bits)) - 1, 8 - self.bits)


def _magnitude(value):
    n = int(abs(value)).bit_length()
    return n, (value if value >= 0 else value + (1 << n) - 1)


def write_jpeg(path, rgb, quality=90, subsample=True, progressive=False, restart=0, gray=False):
    rgb = np.asarray(rgb, dtype=np.float64)
    height, width = rgb.shape[:2]
    if gray:
        planes = [0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]]
        sampling = [(1, 1)]
    else:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        planes = [0.299 * r + 0.587 * g + 0.114 * b,
                  -0.168736 * r - 0.331264 * g + 0.5 * b + 128,
                  0.5 * r - 0.418688 * g - 0.081312 * b + 128]
        sampling = [(2, 2), (1, 1), (1, 1)] if subsample else [(1, 1)] * 3
    hmax, vmax = max(s[0] for s in sampling), max(s[1] for s in sampling)
    mcux, mcuy = -(-width // (8 * hmax)), -(-height // (8 * vmax))
    tables = [_quant(LUMA, quality), _quant(CHROMA, quality)]
    dct = _dct_matrix()
    components = []
    for c, plane in enumerate(planes):
        h, v = sampling[c]
        full = np.pad(plane, ((0, mcuy * 8 * vmax - height), (0, mcux * 8 * hmax - width)), mode="edge")
        if (h, v) != (hmax, vmax):
            fy, fx = vmax // v, hmax // h
            full = full.reshape(full.shape[0] // fy, fy, full.shape[1] // fx, fx).mean(axis=(1, 3))
        by, bx = full.shape[0] // 8, full.shape[1] // 8
        blocks = (full - 128).reshape(by, 8, bx, 8).transpose(0, 2, 1, 3)
        coef = np.einsum("ux,abxy,vy->abuv", dct, blocks, dct).reshape(by, bx, 64)
        q = tables[min(c, 1)]
        coef = np.round(coef / q.reshape(64)).astype(np.int64)[..., ZIGZAG]
        cw, ch = -(-width * h // hmax), -(-height * v // vmax)
        components.append(dict(h=h, v=v, coef=coef, table=min(c, 1), bw=-(-cw // 8), bh=-(-ch // 8)))

    def blocks_of(scan):
        """(component index, block) in scan order, with restart boundaries."""
        units = []
        if len(scan) == 1:
            comp = components[scan[0]]
            for y in range(comp["bh"]):
                for x in range(comp["bw"]):
                    units.append([(scan[0], comp["coef"][y, x])])
        else:
            for my in range(mcuy):
                for mx in range(mcux):
                    unit = []
                    for c in scan:
                        comp = components[c]
                        for v in range(comp["v"]):
                            for h in range(comp["h"]):
                                unit.append((c, comp["coef"][my * comp["v"] + v, mx * comp["h"] + h]))
                    units.append(unit)
        return units

    def encode_scan(scan, ss, se, ah, al):
        """Returns (events, dc counts per table, ac counts per table)."""
        events, dc_counts, ac_counts = [], [{}, {}], [{}, {}]
        state = dict(eobrun=0, pending=[], pred={c: 0 for c in scan})

        def symbol(kind, table, value):
            counts = dc_counts if kind == "dc" else ac_counts
            counts[table][value] = counts[table].get(value, 0) + 1
            events.append((kind, table, value))

        def bits(value, n):
            if n:
                events.append(("bits", value, n))

        def flush_eob(table):
            if state["eobrun"]:
                n = state["eobrun"].bit_length() - 1
                symbol("ac", table, n << 4)
                bits(state["eobrun"] - (1 << n), n)
                state["eobrun"] = 0
            for bit in state["pending"]:
                bits(bit, 1)
            state["pending"] = []

        units = blocks_of(scan)
        for index, unit in enumerate(units):
            if restart and index and index % restart == 0:
                if ss > 0:
                    flush_eob(components[scan[0]]["table"])
                events.append(("restart", (index // restart - 1) % 8))
                state["pred"] = {c: 0 for c in scan}
            for c, block in unit:
                table = components[c]["table"]
                if ss == 0:
                    if ah == 0:
                        value = int(block[0]) >> al
                        diff = value - state["pred"][c]
                        state["pred"][c] = value
                        n, extra = _magnitude(diff)
                        symbol("dc", table, n)
                        bits(extra, n)
                    else:
                        bits((int(block[0]) >> al) & 1, 1)
                    if progressive:
                        continue
                if not progressive:
                    run = 0
                    for k in range(1, 64):
                        value = int(block[k])
                        if value == 0:
                            run += 1
                            continue
                        while run > 15:
                            symbol("ac", table, 0xF0)
                            run -= 16
                        n, extra = _magnitude(value)
                        symbol("ac", table, (run << 4) | n)
                        bits(extra, n)
                        run = 0
                    if run:
                        symbol("ac", table, 0)
                    continue
                if ah == 0:  # AC first
                    run = 0
                    for k in range(ss, se + 1):
                        value = int(block[k])
                        value = (abs(value) >> al) * (1 if value >= 0 else -1)
                        if value == 0:
                            run += 1
                            continue
                        flush_eob(table)
                        while run > 15:
                            symbol("ac", table, 0xF0)
                            run -= 16
                        n, extra = _magnitude(value)
                        symbol("ac", table, (run << 4) | n)
                        bits(extra, n)
                        run = 0
                    if run:
                        state["eobrun"] += 1
                        if state["eobrun"] == 0x7FFF:
                            flush_eob(table)
                    continue
                # AC refinement (libjpeg encode_mcu_AC_refine).
                absolute = [abs(int(block[k])) >> al for k in range(64)]
                eob = max([k for k in range(ss, se + 1) if absolute[k] == 1], default=0)
                run, correction = 0, []
                for k in range(ss, se + 1):
                    value = absolute[k]
                    if value == 0:
                        run += 1
                        continue
                    while run > 15 and k <= eob:
                        flush_eob(table)
                        symbol("ac", table, 0xF0)
                        run -= 16
                        for bit in correction:
                            bits(bit, 1)
                        correction = []
                    if value > 1:
                        correction.append(value & 1)
                        continue
                    flush_eob(table)
                    symbol("ac", table, (run << 4) | 1)
                    bits(1 if block[k] > 0 else 0, 1)
                    for bit in correction:
                        bits(bit, 1)
                    correction, run = [], 0
                if run > 0 or correction:
                    state["eobrun"] += 1
                    state["pending"] += correction
                    if state["eobrun"] == 0x7FFF or len(state["pending"]) > 900:
                        flush_eob(table)
        if ss > 0:
            flush_eob(components[scan[0]]["table"])
        return events, dc_counts, ac_counts

    if progressive:
        luma, rest = [0], list(range(1, len(components)))
        script = [(list(range(len(components))), 0, 0, 0, 1), (luma, 1, 5, 0, 2)]
        script += [([c], 1, 63, 0, 1) for c in rest]
        script += [(luma, 6, 63, 0, 2), (luma, 1, 63, 2, 1), (list(range(len(components))), 0, 0, 1, 0)]
        script += [([c], 1, 63, 1, 0) for c in rest] + [(luma, 1, 63, 1, 0)]
    else:
        script = [(list(range(len(components))), 0, 63, 0, 0)]

    out = bytearray(b"\xff\xd8")
    out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    for i, table in enumerate(tables[:1 if gray else 2]):
        out += b"\xff\xdb" + struct.pack(">HB", 67, i) + bytes(table[ZIGZAG].astype(np.uint8))
    frame = struct.pack(">BHHB", 8, height, width, len(components))
    for c, comp in enumerate(components):
        frame += struct.pack(">BBB", c + 1, comp["h"] << 4 | comp["v"], comp["table"])
    out += (b"\xff\xc2" if progressive else b"\xff\xc0") + struct.pack(">H", 2 + len(frame)) + frame
    if restart:
        out += b"\xff\xdd" + struct.pack(">HH", 4, restart)
    for scan, ss, se, ah, al in script:
        events, dc_counts, ac_counts = encode_scan(scan, ss, se, ah, al)
        codes = {}
        for kind, counts in (("dc", dc_counts), ("ac", ac_counts)):
            for table, used in enumerate(counts):
                if not used:
                    continue
                lengths, symbols, table_codes = _huffman_table(used)
                codes[(kind, table)] = table_codes
                body = struct.pack(">B", (16 if kind == "ac" else 0) | table) + bytes(lengths) + bytes(symbols)
                out += b"\xff\xc4" + struct.pack(">H", 2 + len(body)) + body
        header = struct.pack(">B", len(scan))
        for c in scan:
            table = components[c]["table"]
            header += struct.pack(">BB", c + 1, table << 4 | table)
        header += struct.pack(">BBB", ss, se, ah << 4 | al)
        out += b"\xff\xda" + struct.pack(">H", 2 + len(header)) + header
        writer = _Writer()
        for event in events:
            if event[0] == "bits":
                writer.put(event[1], event[2])
            elif event[0] == "restart":
                writer.flush()
                writer.out += bytes([0xFF, 0xD0 + event[1]])
            else:
                code, size = codes[(event[0], event[1])][event[2]]
                writer.put(code, size)
        writer.flush()
        out += writer.out
    out += b"\xff\xd9"
    open(path, "wb").write(bytes(out))
