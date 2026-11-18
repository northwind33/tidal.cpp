// Dependency-free image decoding for the vision input: PNG, JPEG, BMP and PPM.
#include "tidal/tidal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cctype>
#include <string>

namespace tidal {
namespace {

[[noreturn]] void fail(const std::string & message) { throw std::invalid_argument("Image decoding failed: " + message); }

constexpr int64_t max_pixels = int64_t(1) << 28; // Larger images are rejected before allocation.
void check_size(int64_t width, int64_t height) {
    if (width <= 0 || height <= 0) fail("empty image");
    if (width > 65535 || height > 65535 || width * height > max_pixels) fail("image dimensions are too large");
}
uint8_t over_white(int value, int alpha) { return uint8_t((value * alpha + 255 * (255 - alpha) + 127) / 255); }

// ---------------------------------------------------------------- inflate (RFC 1950/1951)

struct BitReader {
    const uint8_t * data; size_t size, pos = 0; uint32_t bits = 0; int count = 0;
    int bit() {
        if (!count) { if (pos >= size) fail("truncated deflate stream"); bits = data[pos++]; count = 8; }
        const int b = int(bits & 1); bits >>= 1; --count; return b;
    }
    uint32_t take(int n) { uint32_t v = 0; for (int i = 0; i < n; ++i) v |= uint32_t(bit()) << i; return v; }
    void align() { bits = 0; count = 0; }
};

struct Huffman {
    std::array<uint16_t, 16> counts{};
    std::vector<uint16_t> symbols;
    void build(const uint8_t * lengths, int n) {
        counts.fill(0);
        for (int i = 0; i < n; ++i) ++counts[lengths[i]];
        counts[0] = 0;
        std::array<uint16_t, 16> offsets{};
        for (int i = 1; i < 16; ++i) offsets[size_t(i)] = uint16_t(offsets[size_t(i - 1)] + counts[size_t(i - 1)]);
        symbols.assign(size_t(n), 0);
        for (int i = 0; i < n; ++i) if (lengths[i]) symbols[offsets[lengths[i]]++] = uint16_t(i);
    }
    int decode(BitReader & in) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= in.bit();
            const int count = counts[size_t(len)];
            if (code - count < first) return symbols[size_t(index + (code - first))];
            index += count; first += count; first <<= 1; code <<= 1;
        }
        fail("invalid Huffman code in deflate stream");
    }
};

std::vector<uint8_t> inflate_zlib(const uint8_t * data, size_t size, size_t expected) {
    if (size < 2 || (data[0] & 0x0f) != 8 || ((data[0] << 8) | data[1]) % 31 || (data[1] & 0x20)) fail("invalid zlib header");
    BitReader in{data + 2, size - 2};
    std::vector<uint8_t> out;
    out.reserve(expected);
    static const uint16_t length_base[] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static const uint8_t length_extra[] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static const uint16_t dist_base[] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,
                                         4097,6145,8193,12289,16385,24577};
    static const uint8_t dist_extra[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    Huffman fixed_lit, fixed_dist;
    {
        uint8_t l[288];
        for (int i = 0; i < 144; ++i) l[i] = 8;
        for (int i = 144; i < 256; ++i) l[i] = 9;
        for (int i = 256; i < 280; ++i) l[i] = 7;
        for (int i = 280; i < 288; ++i) l[i] = 8;
        fixed_lit.build(l, 288);
        uint8_t d[30]; std::fill(d, d + 30, 5); fixed_dist.build(d, 30);
    }
    for (bool last = false; !last;) {
        last = in.bit();
        const int type = int(in.take(2));
        if (type == 0) {
            in.align();
            if (in.pos + 4 > in.size) fail("truncated stored block");
            const int len = in.data[in.pos] | (in.data[in.pos + 1] << 8), nlen = in.data[in.pos + 2] | (in.data[in.pos + 3] << 8);
            in.pos += 4;
            if ((len ^ 0xffff) != nlen || in.pos + size_t(len) > in.size) fail("invalid stored block");
            out.insert(out.end(), in.data + in.pos, in.data + in.pos + len);
            in.pos += size_t(len);
            continue;
        }
        if (type == 3) fail("invalid deflate block type");
        Huffman dynamic_lit, dynamic_dist;
        const Huffman * lit = &fixed_lit, * dist = &fixed_dist;
        if (type == 2) {
            const int hlit = int(in.take(5)) + 257, hdist = int(in.take(5)) + 1, hclen = int(in.take(4)) + 4;
            static const uint8_t order[] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
            uint8_t code_lengths[19] = {};
            for (int i = 0; i < hclen; ++i) code_lengths[order[i]] = uint8_t(in.take(3));
            Huffman lengths; lengths.build(code_lengths, 19);
            uint8_t l[320] = {};
            for (int n = 0; n < hlit + hdist;) {
                const int sym = lengths.decode(in);
                if (sym < 16) { l[n++] = uint8_t(sym); continue; }
                int repeat = 0; uint8_t value = 0;
                if (sym == 16) { if (!n) fail("invalid code length repeat"); value = l[n - 1]; repeat = 3 + int(in.take(2)); }
                else if (sym == 17) repeat = 3 + int(in.take(3));
                else repeat = 11 + int(in.take(7));
                if (n + repeat > hlit + hdist) fail("code lengths overflow");
                while (repeat--) l[n++] = value;
            }
            dynamic_lit.build(l, hlit); dynamic_dist.build(l + hlit, hdist);
            lit = &dynamic_lit; dist = &dynamic_dist;
        }
        for (;;) {
            const int sym = lit->decode(in);
            if (sym < 256) { out.push_back(uint8_t(sym)); continue; }
            if (sym == 256) break;
            if (sym > 285) fail("invalid length symbol");
            const int length = length_base[sym - 257] + int(in.take(length_extra[sym - 257]));
            const int dsym = dist->decode(in);
            if (dsym > 29) fail("invalid distance symbol");
            const size_t distance = dist_base[dsym] + in.take(dist_extra[dsym]);
            if (distance > out.size()) fail("distance beyond output");
            const size_t from = out.size() - distance;
            for (int i = 0; i < length; ++i) out.push_back(out[from + size_t(i)]);
            if (out.size() > expected + (1u << 20)) fail("decompressed data exceeds the image size");
        }
    }
    return out;
}

// ---------------------------------------------------------------- PNG

uint32_t be32(const uint8_t * p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

Image decode_png(const uint8_t * data, size_t size) {
    size_t pos = 8;
    int width = 0, height = 0, depth = 0, color = -1, interlace = 0;
    std::vector<uint8_t> idat, palette, trns;
    bool header = false, end = false;
    while (!end) {
        if (pos + 8 > size) fail("truncated PNG chunk");
        const uint32_t length = be32(data + pos);
        const std::string type(reinterpret_cast<const char *>(data + pos + 4), 4);
        if (length > size - pos - 12) fail("truncated PNG chunk");
        const uint8_t * body = data + pos + 8;
        if (type == "IHDR") {
            if (length != 13) fail("invalid IHDR");
            width = int(be32(body)); height = int(be32(body + 4)); depth = body[8]; color = body[9]; interlace = body[12];
            if (body[10] || body[11] || interlace > 1) fail("unsupported PNG compression, filter or interlace method");
            header = true;
        } else if (type == "PLTE") palette.assign(body, body + length);
        else if (type == "tRNS") trns.assign(body, body + length);
        else if (type == "IDAT") idat.insert(idat.end(), body, body + length);
        else if (type == "IEND") end = true;
        else if (!(type[0] & 0x20)) fail("unknown critical PNG chunk " + type);
        pos += 12 + length;
    }
    if (!header) fail("PNG without IHDR");
    check_size(width, height);
    const int channels = color == 0 ? 1 : color == 2 ? 3 : color == 3 ? 1 : color == 4 ? 2 : color == 6 ? 4 : 0;
    const bool valid_depth = color == 0 ? (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)
        : color == 3 ? (depth == 1 || depth == 2 || depth == 4 || depth == 8) : (depth == 8 || depth == 16);
    if (!channels || !valid_depth) fail("unsupported PNG color type or bit depth");
    if (color == 3 && (palette.empty() || palette.size() % 3)) fail("palette PNG without a valid PLTE");
    const int bits = channels * depth, bpp = std::max(1, bits / 8);
    size_t expected = 0;
    struct Pass { int x0, y0, dx, dy; };
    const std::vector<Pass> passes = interlace ? std::vector<Pass>{{0,0,8,8},{4,0,8,8},{0,4,4,8},{2,0,4,4},{0,2,2,4},{1,0,2,2},{0,1,1,2}}
                                               : std::vector<Pass>{{0,0,1,1}};
    for (const auto & p : passes) {
        const int64_t pw = (width - p.x0 + p.dx - 1) / p.dx, ph = (height - p.y0 + p.dy - 1) / p.dy;
        if (pw > 0 && ph > 0) expected += size_t(ph) * (1 + size_t((pw * bits + 7) / 8));
    }
    const auto raw = inflate_zlib(idat.data(), idat.size(), expected);
    if (raw.size() < expected) fail("truncated PNG image data");
    Image image; image.width = width; image.height = height;
    image.rgb.assign(size_t(width) * height * 3, 0);
    const auto sample = [&](const uint8_t * row, int x, int c) -> int {
        if (depth == 8) return row[x * channels + c];
        if (depth == 16) return row[(x * channels + c) * 2] << 8 | row[(x * channels + c) * 2 + 1];
        const int bit = x * depth;
        return (row[bit / 8] >> (8 - depth - bit % 8)) & ((1 << depth) - 1);
    };
    const int max = (1 << depth) - 1;
    const auto to8 = [&](int v) { return depth == 16 ? v >> 8 : depth == 8 ? v : v * 255 / max; };
    size_t offset = 0;
    std::vector<uint8_t> previous, current;
    for (const auto & p : passes) {
        const int pw = (width - p.x0 + p.dx - 1) / p.dx, ph = (height - p.y0 + p.dy - 1) / p.dy;
        if (pw <= 0 || ph <= 0) continue;
        const size_t stride = size_t((int64_t(pw) * bits + 7) / 8);
        previous.assign(stride, 0);
        for (int y = 0; y < ph; ++y) {
            const int filter = raw[offset];
            current.assign(raw.begin() + long(offset + 1), raw.begin() + long(offset + 1 + stride));
            offset += 1 + stride;
            for (size_t i = 0; i < stride; ++i) {
                const int a = i >= size_t(bpp) ? current[i - size_t(bpp)] : 0, b = previous[i], c = i >= size_t(bpp) ? previous[i - size_t(bpp)] : 0;
                int add = 0;
                switch (filter) {
                    case 0: break;
                    case 1: add = a; break;
                    case 2: add = b; break;
                    case 3: add = (a + b) / 2; break;
                    case 4: { const int pa = std::abs(b - c), pb = std::abs(a - c), pc = std::abs(a + b - 2 * c);
                              add = pa <= pb && pa <= pc ? a : pb <= pc ? b : c; break; }
                    default: fail("invalid PNG filter");
                }
                current[i] = uint8_t(current[i] + add);
            }
            for (int x = 0; x < pw; ++x) {
                int r, g, b, alpha = 255;
                if (color == 3) {
                    const int index = sample(current.data(), x, 0);
                    if (size_t(index) * 3 + 2 >= palette.size()) fail("palette index out of range");
                    r = palette[size_t(index) * 3]; g = palette[size_t(index) * 3 + 1]; b = palette[size_t(index) * 3 + 2];
                    if (size_t(index) < trns.size()) alpha = trns[size_t(index)];
                } else if (color == 0 || color == 4) {
                    const int v = sample(current.data(), x, 0);
                    r = g = b = to8(v);
                    if (color == 4) alpha = to8(sample(current.data(), x, 1));
                    else if (trns.size() >= 2 && v == (trns[0] << 8 | trns[1])) alpha = 0;
                } else {
                    const int rv = sample(current.data(), x, 0), gv = sample(current.data(), x, 1), bv = sample(current.data(), x, 2);
                    r = to8(rv); g = to8(gv); b = to8(bv);
                    if (color == 6) alpha = to8(sample(current.data(), x, 3));
                    else if (trns.size() >= 6 && rv == (trns[0] << 8 | trns[1]) && gv == (trns[2] << 8 | trns[3]) && bv == (trns[4] << 8 | trns[5])) alpha = 0;
                }
                uint8_t * out = image.rgb.data() + (size_t(p.y0 + y * p.dy) * width + size_t(p.x0 + x * p.dx)) * 3;
                if (alpha == 255) { out[0] = uint8_t(r); out[1] = uint8_t(g); out[2] = uint8_t(b); }
                else { out[0] = over_white(r, alpha); out[1] = over_white(g, alpha); out[2] = over_white(b, alpha); }
            }
            previous.swap(current);
        }
    }
    return image;
}

// ---------------------------------------------------------------- JPEG

struct Jpeg {
    Jpeg(const uint8_t * d, size_t n) : data(d), size(n) {}
    const uint8_t * data; size_t size, pos = 0;
    struct Table { std::array<uint8_t, 17> bits{}; std::array<uint8_t, 256> values{}; std::array<int, 18> maxcode{}; std::array<int, 17> valptr{}, mincode{}; std::array<uint16_t, 512> fast{}; bool present = false; };
    Table dc[4], ac[4];
    std::array<std::array<uint16_t, 64>, 4> quant{};
    struct Component { int id, h, v, tq, td = 0, ta = 0, bw = 0, bh = 0, dc_pred = 0; std::vector<int16_t> coefficients; };
    std::vector<Component> components;
    int width = 0, height = 0, hmax = 1, vmax = 1, mcux = 0, mcuy = 0, restart = 0, eobrun = 0;
    bool progressive = false, adobe = false; int transform = -1, scans = 0;
    // Entropy-coded segment reader.
    uint32_t buffer = 0; int bits = 0; bool marker_hit = false;

    uint8_t byte() { if (pos >= size) fail("truncated JPEG"); return data[pos++]; }
    int word() { const int hi = byte(); return hi << 8 | byte(); }
    void fill() {
        while (bits <= 24) {
            int b = 0;
            if (!marker_hit && pos < size) {
                b = data[pos];
                if (b == 0xff) {
                    const int next = pos + 1 < size ? data[pos + 1] : 0;
                    if (next == 0) pos += 2;
                    else { marker_hit = true; b = 0; }
                } else ++pos;
            }
            buffer |= uint32_t(b) << (24 - bits); bits += 8;
        }
    }
    int get_bit() { fill(); const int b = int(buffer >> 31); buffer <<= 1; --bits; return b; }
    int get_bits(int n) { if (!n) return 0; fill(); const int v = int(buffer >> (32 - n)); buffer <<= n; bits -= n; return v; }
    static int extend(int v, int n) { return n && v < (1 << (n - 1)) ? v - (1 << n) + 1 : v; }
    int decode(const Table & t) {
        if (!t.present) fail("missing Huffman table");
        fill();
        if (const uint16_t entry = t.fast[buffer >> 23]) {
            const int len = entry >> 8;
            buffer <<= len; bits -= len;
            return entry & 0xff;
        }
        int code = 0;
        for (int len = 1; len <= 16; ++len) {
            code = (code << 1) | get_bit();
            if (code <= t.maxcode[size_t(len)]) return t.values[size_t(t.valptr[size_t(len)] + code - t.mincode[size_t(len)])];
        }
        fail("invalid JPEG Huffman code");
    }
    void reset_entropy() {
        buffer = 0; bits = 0; marker_hit = false; eobrun = 0;
        for (auto & c : components) c.dc_pred = 0;
    }
    void read_restart() {
        // Skip to the RSTn marker after the entropy-coded data.
        while (pos + 1 < size && !(data[pos] == 0xff && data[pos + 1] >= 0xd0 && data[pos + 1] <= 0xd7)) ++pos;
        if (pos + 1 >= size) fail("missing JPEG restart marker");
        pos += 2;
        reset_entropy();
    }

    void parse_huffman(int length) {
        const size_t end = pos + size_t(length);
        while (pos < end) {
            const int info = byte(), cls = info >> 4, id = info & 15;
            if (cls > 1 || id > 3) fail("invalid Huffman table id");
            Table & t = cls ? ac[id] : dc[id];
            int total = 0;
            for (int i = 1; i <= 16; ++i) { t.bits[size_t(i)] = byte(); total += t.bits[size_t(i)]; }
            if (total > 256) fail("invalid Huffman table");
            for (int i = 0; i < total; ++i) t.values[size_t(i)] = byte();
            int code = 0, k = 0;
            for (int len = 1; len <= 16; ++len) {
                t.valptr[size_t(len)] = k; t.mincode[size_t(len)] = code;
                code += t.bits[size_t(len)]; k += t.bits[size_t(len)];
                t.maxcode[size_t(len)] = t.bits[size_t(len)] ? code - 1 : -1;
                code <<= 1;
            }
            // 9-bit prefix table: entry = length << 8 | symbol (zero: longer code).
            t.fast.fill(0);
            for (int len = 1; len <= 9; ++len)
                for (int i = 0; i < t.bits[size_t(len)]; ++i) {
                    const int c = t.mincode[size_t(len)] + i, symbol = t.values[size_t(t.valptr[size_t(len)] + i)];
                    const int first = c << (9 - len), count = 1 << (9 - len);
                    for (int j = 0; j < count; ++j) t.fast[size_t(first + j)] = uint16_t(len << 8 | symbol);
                }
            t.present = true;
        }
    }
    void parse_quant(int length) {
        const size_t end = pos + size_t(length);
        static const uint8_t zigzag[64] = {0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
                                           35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};
        while (pos < end) {
            const int info = byte(), precision = info >> 4, id = info & 15;
            if (id > 3 || precision > 1) fail("invalid quantization table");
            for (int i = 0; i < 64; ++i) quant[size_t(id)][zigzag[i]] = uint16_t(precision ? word() : byte());
        }
    }
    void parse_frame() {
        if (byte() != 8) fail("only 8-bit JPEG samples are supported");
        height = word(); width = word();
        const int count = byte();
        if (count != 1 && count != 3 && count != 4) fail("unsupported JPEG component count");
        check_size(width, height);
        for (int i = 0; i < count; ++i) {
            Component c{};
            c.id = byte(); const int s = byte(); c.h = s >> 4; c.v = s & 15; c.tq = byte();
            if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) fail("invalid JPEG component sampling");
            components.push_back(std::move(c));
        }
        for (auto & c : components) { hmax = std::max(hmax, c.h); vmax = std::max(vmax, c.v); }
        mcux = (width + 8 * hmax - 1) / (8 * hmax); mcuy = (height + 8 * vmax - 1) / (8 * vmax);
        for (auto & c : components) {
            c.bw = mcux * c.h; c.bh = mcuy * c.v;
            c.coefficients.assign(size_t(c.bw) * c.bh * 64, 0);
        }
    }
    // Coefficients of block (bx, by) of a component, in zigzag order index k.
    static const uint8_t * zigzag() {
        static const uint8_t z[64] = {0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
                                      35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};
        return z;
    }
    void decode_block(Component & c, int16_t * block, int ss, int se, int ah, int al) {
        const uint8_t * z = zigzag();
        if (!progressive) {
            const int t = decode(dc[c.td]);
            const int diff = t ? extend(get_bits(t), t) : 0;
            c.dc_pred += diff; block[0] = int16_t(c.dc_pred);
            for (int k = 1; k < 64;) {
                const int rs = decode(ac[c.ta]), r = rs >> 4, s = rs & 15;
                if (!s) { if (r != 15) break; k += 16; continue; }
                k += r;
                if (k > 63) fail("JPEG coefficient index out of range");
                block[z[k++]] = int16_t(extend(get_bits(s), s));
            }
            return;
        }
        if (ss == 0) { // DC scan
            if (!ah) {
                const int t = decode(dc[c.td]);
                const int diff = t ? extend(get_bits(t), t) : 0;
                c.dc_pred += diff; block[0] = int16_t(c.dc_pred * (1 << al));
            } else if (get_bit()) block[0] = int16_t(block[0] | (1 << al));
            return;
        }
        if (!ah) { // AC first
            if (eobrun) { --eobrun; return; }
            for (int k = ss; k <= se;) {
                const int rs = decode(ac[c.ta]), r = rs >> 4, s = rs & 15;
                if (!s) {
                    if (r < 15) { eobrun = (1 << r) - 1; if (r) eobrun += get_bits(r); break; }
                    k += 16; continue;
                }
                k += r;
                if (k > 63) fail("JPEG coefficient index out of range");
                block[z[k++]] = int16_t(extend(get_bits(s), s) * (1 << al));
            }
            return;
        }
        // AC refinement (successive approximation of earlier coefficients).
        const int p1 = 1 << al, m1 = -1 << al;
        const auto refine = [&](int16_t & coef) {
            if (get_bit() && !(coef & p1)) coef = int16_t(coef >= 0 ? coef + p1 : coef + m1);
        };
        if (eobrun) {
            --eobrun;
            for (int k = ss; k <= se; ++k) if (block[z[k]]) refine(block[z[k]]);
            return;
        }
        int k = ss;
        do {
            const int rs = decode(ac[c.ta]);
            int r = rs >> 4, value = 0;
            if (!(rs & 15)) {
                if (r < 15) { eobrun = (1 << r) - 1; if (r) eobrun += get_bits(r); r = 64; }
            } else value = get_bit() ? p1 : m1;
            while (k <= se) {
                int16_t & coef = block[z[k++]];
                if (coef) refine(coef);
                else { if (!r) { coef = int16_t(value); break; } --r; }
            }
        } while (k <= se);
    }
    void parse_scan() {
        const int count = byte();
        std::vector<Component *> scan;
        for (int i = 0; i < count; ++i) {
            const int id = byte(), tables = byte();
            auto it = std::find_if(components.begin(), components.end(), [&](const Component & c) { return c.id == id; });
            if (it == components.end()) fail("JPEG scan references an unknown component");
            it->td = tables >> 4; it->ta = tables & 15;
            if (it->td > 3 || it->ta > 3) fail("invalid JPEG table selector");
            scan.push_back(&*it);
        }
        const int ss = byte(), se = byte(), a = byte(), ah = a >> 4, al = a & 15;
        if (progressive ? (ss > se || se > 63 || (ss == 0 && se != 0) || (ss > 0 && count != 1)) : (ss != 0 || se != 63))
            fail("invalid JPEG spectral selection");
        reset_entropy();
        int until_restart = restart;
        int64_t units = 0, total = 0;
        const auto restart_check = [&] {
            ++units;
            if (!restart || units == total) return;
            if (--until_restart == 0) { read_restart(); until_restart = restart; }
        };
        if (count == 1) {
            // Non-interleaved: the component's own block grid, clipped to the image.
            Component & c = *scan[0];
            const int cw = int((int64_t(width) * c.h + hmax - 1) / hmax), ch = int((int64_t(height) * c.v + vmax - 1) / vmax);
            const int bx_count = (cw + 7) / 8, by_count = (ch + 7) / 8;
            total = int64_t(bx_count) * by_count;
            for (int by = 0; by < by_count; ++by)
                for (int bx = 0; bx < bx_count; ++bx) {
                    decode_block(c, c.coefficients.data() + (size_t(by) * c.bw + bx) * 64, ss, se, ah, al);
                    restart_check();
                }
        } else {
            total = int64_t(mcux) * mcuy;
            for (int my = 0; my < mcuy; ++my)
                for (int mx = 0; mx < mcux; ++mx) {
                    for (auto * c : scan)
                        for (int v = 0; v < c->v; ++v)
                            for (int h = 0; h < c->h; ++h) {
                                const int bx = mx * c->h + h, by = my * c->v + v;
                                decode_block(*c, c->coefficients.data() + (size_t(by) * c->bw + bx) * 64, ss, se, ah, al);
                            }
                    restart_check();
                }
        }
        // Continue after the entropy-coded segment.
        while (pos + 1 < size && !(data[pos] == 0xff && data[pos + 1] != 0 && !(data[pos + 1] >= 0xd0 && data[pos + 1] <= 0xd7))) ++pos;
    }
    // Separable float IDCT of one dequantized block, level-shifted and clamped.
    static void idct(const float * in, uint8_t * out, int stride) {
        static float table[8][8];
        static bool ready = false;
        if (!ready) {
            for (int x = 0; x < 8; ++x)
                for (int u = 0; u < 8; ++u)
                    table[x][u] = float((u ? 1.0 : std::sqrt(0.5)) * std::cos((2 * x + 1) * u * M_PI / 16) / 2);
            ready = true;
        }
        float tmp[64];
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) {
                float s = 0;
                for (int u = 0; u < 8; ++u) s += table[x][u] * in[y * 8 + u];
                tmp[y * 8 + x] = s;
            }
        for (int x = 0; x < 8; ++x)
            for (int y = 0; y < 8; ++y) {
                float s = 0;
                for (int v = 0; v < 8; ++v) s += table[y][v] * tmp[v * 8 + x];
                const int value = int(std::lround(s + 128.0f));
                out[y * stride + x] = uint8_t(std::clamp(value, 0, 255));
            }
    }
    Image finish() {
        if (components.empty()) fail("JPEG without a frame");
        if (!scans) fail("JPEG without image data");
        // Sample planes at the component resolution (whole blocks).
        std::vector<std::vector<uint8_t>> planes(components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            auto & c = components[i];
            const int pw = c.bw * 8, ph = c.bh * 8;
            planes[i].assign(size_t(pw) * ph, 0);
            float block[64];
            for (int by = 0; by < c.bh; ++by)
                for (int bx = 0; bx < c.bw; ++bx) {
                    const int16_t * coef = c.coefficients.data() + (size_t(by) * c.bw + bx) * 64;
                    for (int k = 0; k < 64; ++k) block[k] = float(coef[k] * quant[size_t(c.tq)][size_t(k)]);
                    idct(block, planes[i].data() + size_t(by) * 8 * pw + size_t(bx) * 8, pw);
                }
        }
        // Upsample to full resolution: fancy (triangle) filters for 2x factors.
        std::vector<std::vector<uint8_t>> full(components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            const auto & c = components[i];
            const int pw = c.bw * 8, sx = hmax / c.h, sy = vmax / c.v;
            if (hmax % c.h || vmax % c.v) fail("unsupported JPEG sampling ratio");
            auto & out = full[i];
            out.assign(size_t(width) * height, 0);
            const auto & p = planes[i];
            const int cw = (width + sx - 1) / sx, ch = (height + sy - 1) / sy;
            const auto at = [&](int x, int y) { return int(p[size_t(std::clamp(y, 0, ch - 1)) * pw + size_t(std::clamp(x, 0, cw - 1))]); };
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    int value;
                    if (sx == 1 && sy == 1) value = at(x, y);
                    else if (sx == 2 && sy == 1) {
                        const int cx = x / 2, other = x & 1 ? cx + 1 : cx - 1;
                        value = (3 * at(cx, y) + at(other, y) + 2) >> 2;
                        if ((x == 0) || (cx == cw - 1 && (x & 1))) value = at(cx, y);
                    } else if (sx == 2 && sy == 2) {
                        const int cx = x / 2, cy = y / 2, oy = y & 1 ? cy + 1 : cy - 1;
                        const auto col = [&](int xx) { return 3 * at(xx, cy) + at(xx, oy); };
                        if (x == 0 || (cx == cw - 1 && (x & 1))) value = (4 * col(cx) + 8) >> 4;
                        else value = (3 * col(cx) + col(x & 1 ? cx + 1 : cx - 1) + 8) >> 4;
                    } else if (sx == 1 && sy == 2) {
                        const int cy = y / 2, oy = y & 1 ? cy + 1 : cy - 1;
                        value = (3 * at(x, cy) + at(x, oy) + 2) >> 2;
                    } else value = at(x / sx, y / sy);
                    out[size_t(y) * width + x] = uint8_t(std::clamp(value, 0, 255));
                }
        }
        Image image; image.width = width; image.height = height;
        image.rgb.resize(size_t(width) * height * 3);
        const size_t n = size_t(width) * height;
        const bool rgb_ids = components.size() == 3 && components[0].id == 'R' && components[1].id == 'G' && components[2].id == 'B';
        for (size_t i = 0; i < n; ++i) {
            uint8_t * o = image.rgb.data() + i * 3;
            if (components.size() == 1) { o[0] = o[1] = o[2] = full[0][i]; continue; }
            float y = full[0][i], cb = float(full[1][i]) - 128, cr = float(full[2][i]) - 128;
            const bool ycc = components.size() == 3 ? !(rgb_ids || (adobe && transform == 0)) : (adobe && transform == 2);
            float r, g, b;
            if (ycc) { r = y + 1.402f * cr; g = y - 0.344136f * cb - 0.714136f * cr; b = y + 1.772f * cb; }
            else { r = y; g = float(full[1][i]); b = float(full[2][i]); }
            if (components.size() == 4) {
                // Adobe CMYK/YCCK is stored inverted: multiply by K.
                const float k = full[3][i];
                r = r * k / 255; g = g * k / 255; b = b * k / 255;
            }
            o[0] = uint8_t(std::clamp(int(std::lround(r)), 0, 255));
            o[1] = uint8_t(std::clamp(int(std::lround(g)), 0, 255));
            o[2] = uint8_t(std::clamp(int(std::lround(b)), 0, 255));
        }
        return image;
    }
    Image run() {
        pos = 2;
        bool frame = false;
        for (;;) {
            if (pos >= size) { if (frame) break; fail("truncated JPEG"); }
            if (byte() != 0xff) continue;
            int marker = byte();
            while (marker == 0xff) marker = byte();
            if (marker == 0xd9) break;                  // EOI
            if (marker >= 0xd0 && marker <= 0xd7) continue;
            const int length = word() - 2;
            if (length < 0 || pos + size_t(length) > size) fail("truncated JPEG segment");
            const size_t next = pos + size_t(length);
            switch (marker) {
                case 0xc0: case 0xc1: frame = true; parse_frame(); break;
                case 0xc2: frame = true; progressive = true; parse_frame(); break;
                case 0xc4: parse_huffman(length); break;
                case 0xdb: parse_quant(length); break;
                case 0xdd: restart = word(); break;
                case 0xda: if (!frame) fail("JPEG scan before frame"); parse_scan(); ++scans; continue;
                case 0xee:
                    if (length >= 12 && std::memcmp(data + pos, "Adobe", 5) == 0) { adobe = true; transform = data[pos + 11]; }
                    break;
                default:
                    if (marker == 0xc3 || (marker >= 0xc5 && marker <= 0xcf && marker != 0xc8 && marker != 0xcc))
                        fail("unsupported JPEG coding (lossless, hierarchical or arithmetic)");
                    break;
            }
            pos = next;
        }
        return finish();
    }
};

// ---------------------------------------------------------------- BMP, PPM

uint32_t le32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint16_t le16(const uint8_t * p) { return uint16_t(p[0] | p[1] << 8); }

Image decode_bmp(const uint8_t * data, size_t size) {
    if (size < 54) fail("truncated BMP");
    const uint32_t offset = le32(data + 10), header = le32(data + 14);
    if (header < 40) fail("unsupported BMP header");
    const int32_t width = int32_t(le32(data + 18)), raw_height = int32_t(le32(data + 22));
    const int bpp = le16(data + 28), compression = int(le32(data + 30));
    const bool top_down = raw_height < 0;
    const int64_t height = top_down ? -int64_t(raw_height) : raw_height;
    check_size(width, height);
    if (!(compression == 0 || (compression == 3 && (bpp == 32 || bpp == 16)))) fail("compressed BMP is not supported");
    if (bpp != 8 && bpp != 24 && bpp != 32) fail("BMP must have 8, 24 or 32 bits per pixel");
    uint32_t masks[4] = {0x00ff0000, 0x0000ff00, 0x000000ff, 0};
    if (compression == 3) {
        if (size < 14 + 40 + 12) fail("truncated BMP masks");
        for (int i = 0; i < 3; ++i) masks[i] = le32(data + 54 + 4 * i);
        if (header >= 56) masks[3] = le32(data + 54 + 12);
    } else if (bpp == 32 && header >= 56) masks[3] = 0;
    const size_t stride = (size_t(width) * bpp / 8 + 3) & ~size_t(3);
    if (offset > size || stride * size_t(height) > size - offset) fail("truncated BMP pixel data");
    std::vector<uint8_t> palette;
    if (bpp == 8) {
        uint32_t colors = le32(data + 46);
        if (!colors) colors = 256;
        const size_t start = 14 + header;
        if (start + size_t(colors) * 4 > offset) fail("invalid BMP palette");
        palette.assign(data + start, data + start + colors * 4);
    }
    const auto channel = [](uint32_t value, uint32_t mask) {
        if (!mask) return 255u;
        int shift = 0; while (!((mask >> shift) & 1)) ++shift;
        const uint32_t max = mask >> shift;
        return ((value & mask) >> shift) * 255u / max;
    };
    Image image; image.width = width; image.height = int(height);
    image.rgb.resize(size_t(width) * size_t(height) * 3);
    for (int64_t y = 0; y < height; ++y) {
        const uint8_t * row = data + offset + stride * size_t(top_down ? y : height - 1 - y);
        for (int x = 0; x < width; ++x) {
            uint8_t * o = image.rgb.data() + (size_t(y) * width + x) * 3;
            if (bpp == 8) {
                const size_t i = size_t(row[x]) * 4;
                if (i + 2 >= palette.size()) fail("BMP palette index out of range");
                o[0] = palette[i + 2]; o[1] = palette[i + 1]; o[2] = palette[i];
            } else if (bpp == 24) { o[0] = row[x * 3 + 2]; o[1] = row[x * 3 + 1]; o[2] = row[x * 3]; }
            else {
                const uint32_t v = le32(row + x * 4);
                const int r = int(channel(v, masks[0])), g = int(channel(v, masks[1])), b = int(channel(v, masks[2]));
                const int a = masks[3] ? int(channel(v, masks[3])) : 255;
                o[0] = over_white(r, a); o[1] = over_white(g, a); o[2] = over_white(b, a);
            }
        }
    }
    return image;
}

Image decode_ppm(const uint8_t * data, size_t size) {
    size_t pos = 2;
    const bool color = data[1] == '6';
    const auto number = [&] {
        for (;;) {
            while (pos < size && std::isspace(data[pos])) ++pos;
            if (pos < size && data[pos] == '#') { while (pos < size && data[pos] != '\n') ++pos; continue; }
            break;
        }
        int64_t value = 0; bool any = false;
        while (pos < size && std::isdigit(data[pos])) { value = value * 10 + (data[pos++] - '0'); any = true; if (value > 1 << 20) fail("invalid PPM header"); }
        if (!any) fail("invalid PPM header");
        return value;
    };
    const int64_t width = number(), height = number(), max = number();
    if (pos >= size || !std::isspace(data[pos])) fail("invalid PPM header");
    ++pos;
    check_size(width, height);
    if (max < 1 || max > 65535) fail("invalid PPM maximum value");
    const int channels = color ? 3 : 1, bytes = max > 255 ? 2 : 1;
    if (size - pos < size_t(width) * size_t(height) * channels * bytes) fail("truncated PPM pixel data");
    Image image; image.width = int(width); image.height = int(height);
    image.rgb.resize(size_t(width) * size_t(height) * 3);
    for (size_t i = 0; i < size_t(width) * size_t(height); ++i)
        for (int c = 0; c < 3; ++c) {
            const size_t at = pos + (i * channels + (color ? c : 0)) * bytes;
            const int64_t v = bytes == 2 ? (data[at] << 8 | data[at + 1]) : data[at];
            image.rgb[i * 3 + c] = uint8_t((v * 255 + max / 2) / max);
        }
    return image;
}

} // namespace

uint64_t Image::hash() const {
    // FNV-1a over the size, then 64-bit words mixed with a multiply-xorshift.
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; h ^= h >> 29; };
    mix(uint64_t(uint32_t(width)) << 32 | uint32_t(height));
    size_t i = 0;
    for (; i + 8 <= rgb.size(); i += 8) { uint64_t v; std::memcpy(&v, rgb.data() + i, 8); h ^= v; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 32; }
    for (; i < rgb.size(); ++i) mix(rgb[i]);
    return h ? h : 1;
}

Image decode_image(const std::string & bytes) {
    const auto * data = reinterpret_cast<const uint8_t *>(bytes.data());
    const size_t size = bytes.size();
    if (size >= 8 && std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0) return decode_png(data, size);
    if (size >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff) { Jpeg jpeg(data, size); return jpeg.run(); }
    if (size >= 2 && data[0] == 'B' && data[1] == 'M') return decode_bmp(data, size);
    if (size >= 2 && data[0] == 'P' && (data[1] == '5' || data[1] == '6')) return decode_ppm(data, size);
    if (size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0)
        fail("WebP is not supported (use PNG, JPEG, BMP or PPM)");
    if (size >= 6 && (std::memcmp(data, "GIF87a", 6) == 0 || std::memcmp(data, "GIF89a", 6) == 0))
        fail("GIF is not supported (use PNG, JPEG, BMP or PPM)");
    fail("unrecognized format (supported: PNG, JPEG, BMP, PPM)");
}

} // namespace tidal
