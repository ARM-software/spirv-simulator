#include <iostream>
#include <cstdint>
#include <bitset>
#include <cassert>

#include "framework/memory_flag_tracker.hpp"


int main() {
    SPIRVSimulator::MemoryFlagTracker mem;

    constexpr std::uint64_t A = 1ull << 0;
    constexpr std::uint64_t B = 1ull << 1;
    constexpr std::uint64_t C = 1ull << 8;
    constexpr std::uint64_t D = 1ull << 16;
    constexpr std::uint64_t E = 1ull << 32;

    mem.write(100, 10, A);     // [100,110) local A
    mem.copy(100, 200, 10);    // [200,210) shares fragment with [100,110)
    mem.copy(100, 300, 10);    // [300,310) shares fragment too

    mem.markLineage(200, 10, C);
    // Because 100/200/300 all reference the same fragment lineage,
    // they all now see P.

    auto q100 = mem.query(100);
    auto q200 = mem.query(200);
    auto q300 = mem.query(300);

    std::cout << "100: " << (q100 ? std::bitset<64>(uint64_t(*q100)) : 0) << "\n";
    std::cout << "200: " << (q200 ? std::bitset<64>(uint64_t(*q200)) : 0) << "\n";
    std::cout << "300: " << (q300 ? std::bitset<64>(uint64_t(*q300)) : 0) << "\n";

    mem.markRange(300, 10, B);
    // Only [300,310) gets local B. Others stay unchanged.

    mem.write(104, 2, D);
    mem.copy(104, 304, 2);

    mem.markLineage(304, 2, E);

    for (const auto& span : mem.queryRange(90, 230)) {
        std::cout << "[" << span.start << ", " << span.end << ") flags=" << std::bitset<64>(uint64_t(span.flags)) << "\n";
    }

    auto ranges = mem.queryAllRangesWithAnyFlags(C);
    for (const auto& r : ranges) {
        std::cout << "Has P: [" << r.start << ", " << r.end << ")\n";
    }
}