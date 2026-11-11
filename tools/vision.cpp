// tidal-vision: encodes images with the vision tower and runs prompts that
// contain them. Used for validation against reference implementations.
#include "args.hpp"
#include "prefill_options.hpp"
#include "report.hpp"
#include "sampling_options.hpp"
#include "vision_options.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
std::vector<std::string> split(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');) if (!item.empty()) out.push_back(item);
    return out;
}
std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(input), {}};
}
void write_file(const std::string & path, const void * data, size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(static_cast<const char *>(data), std::streamsize(bytes));
    if (!output) throw std::runtime_error("Cannot write " + path);
}
void placement_json(std::ostream & out, const std::vector<tidal::TensorPlacement> & list) {
    out << '[';
    for (size_t i = 0; i < list.size(); ++i) {
        const auto & t = list[i];
        if (i) out << ',';
        out << "{\"name\":"; report::string(out, t.name);
        out << ",\"component\":"; report::string(out, t.component);
        out << ",\"role\":"; report::string(out, t.role);
        out << ",\"group\":"; report::string(out, t.group);
        out << ",\"type\":"; report::string(out, t.type);
        out << ",\"shape\":"; report::array(out, t.shape);
        out << ",\"bytes\":" << t.bytes << ",\"resident\":" << (t.resident ? "true" : "false");
        out << ",\"memory\":"; report::string(out, t.memory);
        out << ",\"compute\":"; report::string(out, t.compute);
        out << '}';
    }
    out << ']';
}
} // namespace

int main(int argc, char ** argv) {
    try {
        Args args(argc, argv, vision_options(prefill_options(sampling_options({"--help", "--model", "--backend", "--device",
            "--threads", "--context", "--images", "--tokens", "--image-token", "--steps", "--json", "--dump-embeddings",
            "--dump-resized", "--placement", "--cached", "--repeat"}))));
        if (!args.get("--help").empty()) {
            std::cout << "tidal-vision --model MODEL.gguf --vision-model MMPROJ.gguf --images a.png[,b.jpg]\n"
                         "  [--tokens 1,2,IMG,3] [--image-token ID] [--steps 0] [--logits 0|1] [--cached 0|1] [--repeat 1]\n"
                         "  [--json trace.json] [--dump-embeddings PREFIX] [--dump-resized PREFIX] [--placement 0|1]\n"
                         "  [--backend cuda|cpu] [--device 0] [--threads 4] [--context N]\n"
                         "Decodes and encodes each image (PREFIX.N.f32 holds embedding N, PREFIX.N.rgb the resized pixels).\n"
                         "With --tokens, each IMG expands to the image's placeholder tokens and the prompt is\n"
                         "prefilled (prefill_cached with --cached 1), followed by --steps greedy decode steps.\n";
            vision_help(std::cout);
            prefill_help(std::cout);
            return 0;
        }
        if (args.get("--model").empty()) {
            // Decode only: dump each image's RGB pixels (PREFIX.N.input.rgb).
            const auto prefix = args.get("--dump-resized");
            if (prefix.empty()) throw std::invalid_argument("--model is required (or --dump-resized alone to decode images)");
            std::ostringstream out;
            out << "{\"images\":[";
            const auto paths = split(args.get("--images"));
            for (size_t i = 0; i < paths.size(); ++i) {
                if (i) out << ',';
                try {
                    const auto image = tidal::decode_image(read_file(paths[i]));
                    write_file(prefix + "." + std::to_string(i) + ".input.rgb", image.rgb.data(), image.rgb.size());
                    out << "{\"width\":" << image.width << ",\"height\":" << image.height << '}';
                } catch (const std::invalid_argument & e) {
                    out << "{\"error\":"; report::string(out, e.what()); out << '}';
                }
            }
            out << "]}\n";
            std::cout << out.str();
            return 0;
        }
        tidal::Options options;
        options.backend = args.get("--backend", "cuda");
        options.device = args.integer("--device", 0);
        options.threads = args.integer("--threads", 4);
        options.context = args.integer("--context", 0);
        options.sampling = sampling_config(args);
        configure_prefill(args, options);
        configure_vision(args, options);
        if (options.vision_model.empty()) throw std::invalid_argument("--vision-model is required");
        const auto load_start = std::chrono::steady_clock::now();
        tidal::Engine engine(args.get("--model"), options);
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
        std::vector<std::shared_ptr<const tidal::Image>> images;
        for (const auto & path : split(args.get("--images"))) images.push_back(std::make_shared<tidal::Image>(tidal::decode_image(read_file(path))));
        std::ostringstream out;
        out << std::setprecision(9);
        out << "{\"load_ms\":" << load_ms << ",\"backend\":"; report::string(out, engine.backend_name());
        const auto info = engine.vision_info();
        out << ",\"vision\":{\"placement\":"; report::string(out, info.placement);
        out << ",\"device\":"; report::string(out, info.device);
        out << ",\"layers\":" << info.layers << ",\"hidden\":" << info.hidden << ",\"output\":" << info.output
            << ",\"min_tokens\":" << info.min_tokens << ",\"max_tokens\":" << info.max_tokens
            << ",\"parameters\":" << info.parameters << ",\"weight_bytes\":" << info.weight_bytes << "}";
        out << ",\"images\":[";
        const auto dump = args.get("--dump-embeddings"), resized = args.get("--dump-resized");
        for (size_t i = 0; i < images.size(); ++i) {
            const auto & image = *images[i];
            if (i) out << ',';
            out << "{\"width\":" << image.width << ",\"height\":" << image.height << ",\"tokens\":"
                << engine.image_tokens(image.width, image.height) << ",\"hash\":\"" << std::hex << image.hash() << std::dec << '"';
            if (!resized.empty()) write_file(resized + "." + std::to_string(i) + ".input.rgb", image.rgb.data(), image.rgb.size());
            if (!dump.empty()) {
                const auto start = std::chrono::steady_clock::now();
                const auto embedding = engine.encode_image(image);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                write_file(dump + "." + std::to_string(i) + ".f32", embedding.values.data(), embedding.values.size() * sizeof(float));
                out << ",\"grid\":[" << embedding.grid_width << ',' << embedding.grid_height << "],\"pixels\":["
                    << embedding.pixel_width << ',' << embedding.pixel_height << "],\"encode_ms\":" << ms;
            }
            out << '}';
        }
        out << ']';
        const auto token_text = args.get("--tokens");
        if (!token_text.empty()) {
            const int placeholder = args.integer("--image-token", engine.config().image_token);
            if (placeholder < 0) throw std::invalid_argument("The model declares no image token; pass --image-token");
            std::vector<int32_t> tokens;
            std::vector<tidal::ImageSpan> spans;
            size_t next = 0;
            for (const auto & item : split(token_text)) {
                if (item == "IMG") {
                    if (next >= images.size()) throw std::invalid_argument("More IMG markers than --images");
                    const auto & image = images[next++];
                    spans.push_back({int(tokens.size()), image, nullptr});
                    tokens.insert(tokens.end(), size_t(engine.image_tokens(image->width, image->height)), placeholder);
                } else tokens.push_back(number(item));
            }
            const bool capture = args.integer("--logits", 0) != 0, cached = args.integer("--cached", 0) != 0;
            const int repeat = args.integer("--repeat", 1), steps = args.integer("--steps", 0);
            out << ",\"prompt_tokens\":" << tokens.size() << ",\"image_offsets\":[";
            for (size_t i = 0; i < spans.size(); ++i) out << (i ? "," : "") << spans[i].offset;
            out << "],\"runs\":[";
            for (int r = 0; r < repeat; ++r) {
                if (!cached) engine.reset();
                const auto before = engine.statistics();
                const auto start = std::chrono::steady_clock::now();
                const auto trace = cached ? engine.prefill_cached(tokens, spans, capture) : engine.prefill(tokens, spans, capture);
                const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                const auto & stats = engine.statistics();
                if (r) out << ',';
                out << "{\"prefill_ms\":" << prefill_ms << ",\"evaluated\":" << trace.size()
                    << ",\"reused\":" << stats.prompt_cache_reused_tokens - (cached ? before.prompt_cache_reused_tokens : 0)
                    << ",\"images_encoded\":" << stats.images_encoded - (cached ? before.images_encoded : 0)
                    << ",\"vision_ms\":" << stats.vision_ms - (cached ? before.vision_ms : 0);
                if (capture) {
                    out << ",\"logits\":[";
                    for (size_t i = 0; i < trace.size(); ++i) { if (i) out << ','; report::array(out, trace[i].logits); }
                    out << ']';
                }
                out << ",\"generated\":[";
                const auto decode_start = std::chrono::steady_clock::now();
                std::vector<std::vector<float>> step_logits;
                for (int s = 0; s < steps; ++s) {
                    if (capture) step_logits.push_back(engine.logits());
                    const auto token = engine.sample();
                    out << (s ? "," : "") << token;
                    if (s + 1 < steps) engine.decode(token, false);
                }
                const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
                out << "],\"decode_ms\":" << decode_ms;
                if (capture) {
                    out << ",\"step_logits\":[";
                    for (size_t i = 0; i < step_logits.size(); ++i) { if (i) out << ','; report::array(out, step_logits[i]); }
                    out << ']';
                }
                out << '}';
            }
            out << ']';
        }
        if (args.integer("--placement", 0)) { out << ",\"placement\":"; placement_json(out, engine.tensor_placements()); }
        out << "}\n";
        const auto json = args.get("--json");
        if (json.empty()) std::cout << out.str();
        else { std::ofstream file(json); file << out.str(); if (!file) throw std::runtime_error("Cannot write " + json); }
        return 0;
    } catch (const std::exception & e) { std::cerr << "tidal-vision: " << e.what() << '\n'; return 1; }
}
