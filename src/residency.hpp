#pragma once
#include "internal.hpp"
#include <algorithm>
#include <numeric>
#include <span>

namespace tidal {
// Caller skips groups already absent and protects every group referenced by the
// active graph. Inactive sublayers of the current layer are the last resort.
inline std::vector<int> weight_eviction_order(const std::vector<WeightGroup> & groups,
                                             int current_layer, std::span<const int> protected_groups) {
    std::vector<int> result;
    for (size_t i = 0; i < groups.size(); ++i)
        if (std::find(protected_groups.begin(), protected_groups.end(), int(i)) == protected_groups.end())
            result.push_back(int(i));
    const auto category = [&](int id) {
        const int layer = groups[id].layer;
        return layer < current_layer ? 0 : layer > current_layer ? 1 : 2;
    };
    std::stable_sort(result.begin(), result.end(), [&](int a, int b) {
        if (category(a) != category(b)) return category(a) < category(b);
        return category(a) == 1 ? groups[a].layer > groups[b].layer : groups[a].layer < groups[b].layer;
    });
    return result;
}
} // namespace tidal
