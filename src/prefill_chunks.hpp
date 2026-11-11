#pragma once
#include "prefill_memory.hpp"
#include <algorithm>

namespace tidal {
// Only execute is retryable. Completion/consumer work is deliberately outside
// the catch: a consumer may already have changed KV or recurrent state.
template<class Execute, class Complete, class Reduce>
void visit_prefill_chunks(int count, int width, bool automatic,
                          Execute execute, Complete complete, Reduce reduce) {
    if (count < 0 || width < 1) throw std::invalid_argument("Invalid prefill chunk range");
    for (int first = 0; first < count;) {
        const int n = std::min(width, count-first);
        try {
            execute(first, n);
        } catch (const PrefillAllocationError &) {
            if (!automatic || n == 1) throw;
            width = std::max(1, n/2);
            reduce(first, n, width);
            continue;
        }
        complete(first, n);
        first += n;
    }
}
} // namespace tidal
