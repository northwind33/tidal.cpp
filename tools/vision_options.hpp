#pragma once
#include "args.hpp"
#include "tidal/tidal.hpp"
#include <ostream>

inline std::vector<std::string> vision_options(std::vector<std::string> names) {
    for (const char * name : {"--vision-model", "--vision-placement", "--image-min-tokens", "--image-max-tokens"}) names.emplace_back(name);
    return names;
}
inline void configure_vision(const Args & args, tidal::Options & options) {
    options.vision_model = args.get("--vision-model");
    const auto placement = args.get("--vision-placement", "auto");
    if (placement == "auto") options.vision_placement = tidal::VisionPlacement::Auto;
    else if (placement == "gpu") options.vision_placement = tidal::VisionPlacement::GPU;
    else if (placement == "cpu") options.vision_placement = tidal::VisionPlacement::CPU;
    else throw std::invalid_argument("--vision-placement must be auto, gpu or cpu");
    options.image_min_tokens = args.integer("--image-min-tokens", 0);
    options.image_max_tokens = args.integer("--image-max-tokens", 0);
    if (options.vision_model.empty() && (placement != "auto" || options.image_min_tokens || options.image_max_tokens))
        throw std::invalid_argument("Vision options require --vision-model");
}
inline void vision_help(std::ostream & out) {
    out << "  [--vision-model MMPROJ.gguf] [--vision-placement auto|gpu|cpu]\n"
           "  [--image-min-tokens 8] [--image-max-tokens 4096]\n"
           "The vision tower (a Qwen3-VL mmproj GGUF) keeps its weights in VRAM and runs on the GPU\n"
           "(gpu), or keeps them in system memory and runs on CPU threads (cpu); auto follows --backend.\n"
           "Images are resized so that their token count (one per 32x32 pixels) stays within the limits.\n";
}
