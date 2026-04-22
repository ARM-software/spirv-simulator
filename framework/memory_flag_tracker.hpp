#pragma once

#ifndef ARM_SPIRV_SIMULATOR_MEMORY_FLAG_TRACKER_HPP
#define ARM_SPIRV_SIMULATOR_MEMORY_FLAG_TRACKER_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

// Enables historic tracking of every memory operation
// Can consume quite a bit of memory for long runs, but is useful for debugging
#define ENABLE_HISTORY 0

namespace SPIRVSimulator
{

class MemoryFlagTracker {
public:
    using Address = std::uint64_t;
    using Size = std::uint64_t;
    using Offset = std::uint64_t;
    using Flags = std::uint64_t;
    using FragmentId = std::uint64_t;

    struct Range {
        Address start;
        Address end; // exclusive

        Range() : start(0), end(0) {}
        Range(Address s, Address e) : start(s), end(e) {}
    };

    struct FlagSpan {
        Address start;
        Address end; // exclusive
        Flags flags;

        FlagSpan() : start(0), end(0), flags(0) {}
        FlagSpan(Address s, Address e, Flags f) : start(s), end(e), flags(f) {}
    };

    enum class OpKind {
        Write,
        Copy,
        MarkLocal,
        MarkLineage
    };

    enum class MappingOrigin {
        WriteRoot,
        Copy
    };

    struct QueryResult {
        Address address;
        Address root_address;
        Flags flags;
        FragmentId fragment_id;
        MappingOrigin origin;

        QueryResult()
            : address(0), root_address(0), flags(0), fragment_id(0), origin(MappingOrigin::WriteRoot) {}

        QueryResult(Address a, Address b, Flags f, FragmentId frag, MappingOrigin o)
            : address(a), root_address(b), flags(f), fragment_id(frag), origin(o) {}
    };

    struct DetailedFlagSpan {
        Address start;
        Address end; // exclusive
        Flags flags;
        FragmentId fragment_id;
        MappingOrigin origin;

        DetailedFlagSpan()
            : start(0), end(0), flags(0), fragment_id(0),
            origin(MappingOrigin::WriteRoot) {}

        DetailedFlagSpan(Address s, Address e, Flags f, FragmentId frag, MappingOrigin o)
            : start(s), end(e), flags(f), fragment_id(frag), origin(o) {}
    };

    struct HistoryEntry {
        OpKind kind;
        Address dst;
        Size size;
        Flags flags;
        Address src;

        HistoryEntry(OpKind k, Address d, Size sz, Flags fl, Address s = 0)
            : kind(k), dst(d), size(sz), flags(fl), src(s) {}
    };

    MemoryFlagTracker() : next_fragment_id_(1) {}

    // Overwrite [addr, addr+size) with a fresh fragment and local flags.
    void write(Address addr, Size size, Flags flags) {
        if (size == 0) {
            return;
        }

        const Address end = checkedEnd(addr, size);
#if ENABLE_HISTORY
        history_.emplace_back(OpKind::Write, addr, size, flags, 0);
#endif

        eraseLiveRange(addr, end);

        const FragmentId frag = createFragment(size, addr);
        insertLiveSpan(LiveSpan(addr, end, frag, 0, flags, MappingOrigin::WriteRoot));
        coalesceLive();
    }

    // Copy currently visible mappings from [src, src+size) to [dst, dst+size).
    //
    // Important:
    // - The destination shares the same underlying fragment identities as the source.
    // - Local flags are copied from the visible source spans.
    // - Shared lineage flags are not duplicated locally; they remain shared via fragment identity.
    // - If the src contains empty ranges (areas not previously covered by a write) they are initialized tih 0x0 flags
    void copy(Address src, Address dst, Size size) {
        if (size == 0) {
            return;
        }

        const Address srcEnd = checkedEnd(src, size);
        const Address dstEnd = checkedEnd(dst, size);

        // Ensure the entire source range exists.
        // Any gaps are materialized as fresh writes with local flags = 0.
        ensureRangeExists(src, srcEnd);

        // Snapshot after materializing missing source gaps, before overwriting dst.
        const std::vector<ResolvedLiveSpan> source = resolveLiveMappings(src, size);

#if ENABLE_HISTORY
        history_.emplace_back(OpKind::Copy, dst, size, 0, src);
#endif
        eraseLiveRange(dst, dstEnd);

        for (const ResolvedLiveSpan& piece : source) {
            const Offset len = piece.abs_end - piece.abs_start;
            const Address newStart = dst + (piece.abs_start - src);
            const Address newEnd = newStart + len;

            insertLiveSpan(LiveSpan(
                newStart,
                newEnd,
                piece.fragment_id,
                piece.fragment_offset,
                piece.local_flags,
                MappingOrigin::Copy));
        }

        coalesceLive();
    }

    // OR flags into the currently visible live spans only.
    // This does NOT propagate to related copies.
    void markRange(Address addr, Size size, Flags flags) {
        if (size == 0 || flags == 0) {
            return;
        }

        const Address end = checkedEnd(addr, size);
#if ENABLE_HISTORY
        history_.emplace_back(OpKind::MarkLocal, addr, size, flags, 0);
#endif

        splitLiveAt(addr);
        splitLiveAt(end);

        auto it = live_.lower_bound(addr);
        while (it != live_.end() && it->second.start < end) {
            it->second.local_flags |= flags;
            ++it;
        }

        coalesceLive();
    }

    // OR flags into the underlying fragment lineage for the currently visible range.
    // This propagates backward and forward across all copies that share the same fragment slice.
    void markLineage(Address addr, Size size, Flags flags) {
        if (size == 0 || flags == 0) {
            return;
        }

        checkedEnd(addr, size);

#if ENABLE_HISTORY
        history_.emplace_back(OpKind::MarkLineage, addr, size, flags, 0);
#endif
        const std::vector<ResolvedLiveSpan> pieces = resolveLiveMappings(addr, size);
        for (const ResolvedLiveSpan& piece : pieces) {
            Fragment& frag = getFragment(piece.fragment_id);
            frag.addSharedFlags(
                piece.fragment_offset,
                piece.fragment_offset + (piece.abs_end - piece.abs_start),
                flags);
        }
    }

    // Effective flags at a single address, or nullopt if unmapped.
    std::optional<Flags> query(Address addr) const {
        auto it = live_.upper_bound(addr);
        if (it == live_.begin()) {
            return std::nullopt;
        }

        --it;
        const LiveSpan& live = it->second;
        if (addr < live.start || addr >= live.end) {
            return std::nullopt;
        }

        const Offset off = live.fragment_offset + (addr - live.start);
        const Fragment& frag = getFragmentConst(live.fragment_id);

        return live.local_flags | frag.sharedFlagsAt(off);
    }

    // Effective flags + origin details at a single address, or nullopt if unmapped.
    std::optional<QueryResult> queryDetailed(Address addr) const {
        auto it = live_.upper_bound(addr);
        if (it == live_.begin()) {
            return std::nullopt;
        }

        --it;
        const LiveSpan& live = it->second;
        if (addr < live.start || addr >= live.end) {
            return std::nullopt;
        }

        const Offset off = live.fragment_offset + (addr - live.start);
        const Fragment& frag = getFragmentConst(live.fragment_id);
        const Flags effective = live.local_flags | frag.sharedFlagsAt(off);

        return QueryResult(addr, frag.source_write_address(), effective, live.fragment_id, live.origin);
    }

    // Returns visible flagged spans over [addr, addr+size).
    // Unmapped gaps are omitted.
    std::vector<FlagSpan> queryRange(Address addr, Size size) const {
        std::vector<FlagSpan> out;
        if (size == 0) {
            return out;
        }

        const Address end = checkedEnd(addr, size);
        auto it = live_.upper_bound(addr);
        if (it != live_.begin()) {
            --it;
        }

        while (it != live_.end()) {
            const LiveSpan& live = it->second;

            if (live.end <= addr) {
                ++it;
                continue;
            }
            if (live.start >= end) {
                break;
            }

            const Address absStart = std::max(addr, live.start);
            const Address absEnd = std::min(end, live.end);

            const Offset fragStart = live.fragment_offset + (absStart - live.start);
            const Offset fragEnd = live.fragment_offset + (absEnd - live.start);

            const Fragment& frag = getFragmentConst(live.fragment_id);
            const auto sharedPieces = frag.queryRange(fragStart, fragEnd);

            for (const auto& sp : sharedPieces) {
                const Address pieceStart = live.start + (sp.start - live.fragment_offset);
                const Address pieceEnd = live.start + (sp.end - live.fragment_offset);
                const Flags flags = live.local_flags | sp.flags;

                appendMerged(out, FlagSpan(pieceStart, pieceEnd, flags));
            }

            ++it;
        }

        return out;
    }

    std::vector<DetailedFlagSpan> queryRangeDetailed(Address addr, Size size) const {
        std::vector<DetailedFlagSpan> out;
        if (size == 0) {
            return out;
        }

        const Address end = checkedEnd(addr, size);
        auto it = live_.upper_bound(addr);
        if (it != live_.begin()) {
            --it;
        }

        while (it != live_.end()) {
            const LiveSpan& live = it->second;

            if (live.end <= addr) {
                ++it;
                continue;
            }
            if (live.start >= end) {
                break;
            }

            const Address absStart = std::max(addr, live.start);
            const Address absEnd = std::min(end, live.end);

            const Offset fragStart = live.fragment_offset + (absStart - live.start);
            const Offset fragEnd = live.fragment_offset + (absEnd - live.start);

            const Fragment& frag = getFragmentConst(live.fragment_id);
            const auto sharedPieces = frag.queryRange(fragStart, fragEnd);

            for (const auto& sp : sharedPieces) {
                const Address pieceStart = live.start + (sp.start - live.fragment_offset);
                const Address pieceEnd = live.start + (sp.end - live.fragment_offset);
                const Flags flags = live.local_flags | sp.flags;

                appendMerged(out, DetailedFlagSpan(
                    pieceStart,
                    pieceEnd,
                    flags,
                    live.fragment_id,
                    live.origin));
            }

            ++it;
        }

        return out;
    }

    // Return all currently visible ranges where ANY bit in mask is set.
    std::vector<Range> queryAllRangesWithAnyFlags(Flags mask) const {
        return queryAllRangesByPredicate(
            [mask](Flags f) { return (f & mask) != 0; });
    }

    // Return all currently visible ranges where ALL bits in mask are set.
    std::vector<Range> queryAllRangesWithAllFlags(Flags mask) const {
        return queryAllRangesByPredicate(
            [mask](Flags f) { return (f & mask) == mask; });
    }

    // Returns an empty vector if ENABLE_HISTORY is 0
    const std::vector<HistoryEntry>& history() const {
        return history_;
    }

private:
    struct SharedFlagSpan {
        Offset start;
        Offset end; // exclusive
        Flags flags;

        SharedFlagSpan() : start(0), end(0), flags(0) {}
        SharedFlagSpan(Offset s, Offset e, Flags f) : start(s), end(e), flags(f) {}
    };

    class Fragment {
    public:
        Fragment() : id_(0), length_(0), source_write_address_(0) {}
        Fragment(FragmentId id, Offset length, Address source_write_address) : id_(id), length_(length), source_write_address_(source_write_address) {}

        FragmentId id() const { return id_; }
        Offset length() const { return length_; }
        Address source_write_address() const { return source_write_address_; }

        Flags sharedFlagsAt(Offset off) const {
            if (off >= length_) {
                throw std::out_of_range("fragment offset out of range");
            }

            auto it = shared_.upper_bound(off);
            if (it == shared_.begin()) {
                return 0;
            }

            --it;
            if (off >= it->second.start && off < it->second.end) {
                return it->second.flags;
            }
            return 0;
        }

        struct QuerySpan {
            Offset start;
            Offset end;
            Flags flags;

            QuerySpan() : start(0), end(0), flags(0) {}
            QuerySpan(Offset s, Offset e, Flags f) : start(s), end(e), flags(f) {}
        };

        std::vector<QuerySpan> queryRange(Offset start, Offset end) const {
            if (start > end || end > length_) {
                throw std::out_of_range("fragment query out of range");
            }

            std::vector<QuerySpan> out;
            if (start == end) {
                return out;
            }

            Offset pos = start;

            auto it = shared_.upper_bound(start);
            if (it != shared_.begin()) {
                --it;
            }

            while (pos < end) {
                while (it != shared_.end() && it->second.end <= pos) {
                    ++it;
                }

                if (it == shared_.end() || it->second.start > pos) {
                    const Offset nextBoundary =
                        (it == shared_.end()) ? end : std::min(end, it->second.start);
                    appendMerged(out, QuerySpan(pos, nextBoundary, 0));
                    pos = nextBoundary;
                    continue;
                }

                const Offset segStart = std::max(pos, it->second.start);
                const Offset segEnd = std::min(end, it->second.end);
                appendMerged(out, QuerySpan(segStart, segEnd, it->second.flags));
                pos = segEnd;
            }

            return out;
        }

        void addSharedFlags(Offset start, Offset end, Flags flags) {
            if (flags == 0 || start >= end) {
                return;
            }
            if (end > length_) {
                throw std::out_of_range("fragment mark out of range");
            }

            splitAt(start);
            splitAt(end);

            auto it = shared_.lower_bound(start);
            Offset pos = start;

            while (pos < end) {
                if (it == shared_.end() || it->second.start > pos) {
                    const Offset gapEnd =
                        (it == shared_.end()) ? end : std::min(end, it->second.start);
                    shared_.emplace(pos, SharedFlagSpan(pos, gapEnd, flags));
                    pos = gapEnd;
                    continue;
                }

                if (it->second.start == pos) {
                    it->second.flags |= flags;
                    pos = it->second.end;
                    ++it;
                    continue;
                }

                throw std::logic_error("unexpected fragment shared-flag state");
            }

            coalesce();
        }

    private:
        FragmentId id_;
        Offset length_;
        Address source_write_address_;
        std::map<Offset, SharedFlagSpan> shared_;

        void splitAt(Offset pos) {
            if (pos == 0 || pos >= length_) {
                return;
            }

            auto it = shared_.upper_bound(pos);
            if (it == shared_.begin()) {
                return;
            }

            --it;
            const SharedFlagSpan span = it->second;
            if (pos <= span.start || pos >= span.end) {
                return;
            }

            it->second.end = pos;
            shared_.emplace(pos, SharedFlagSpan(pos, span.end, span.flags));
        }

        void coalesce() {
            if (shared_.empty()) {
                return;
            }

            auto it = shared_.begin();
            while (it != shared_.end()) {
                auto next = std::next(it);
                if (next == shared_.end()) {
                    break;
                }

                if (it->second.end == next->second.start &&
                    it->second.flags == next->second.flags) {
                    it->second.end = next->second.end;
                    shared_.erase(next);
                } else {
                    ++it;
                }
            }
        }

        static void appendMerged(std::vector<QuerySpan>& out, const QuerySpan& span) {
            if (span.start >= span.end) {
                return;
            }

            if (!out.empty() &&
                out.back().end == span.start &&
                out.back().flags == span.flags) {
                out.back().end = span.end;
            } else {
                out.push_back(span);
            }
        }
    };

    struct LiveSpan {
        Address start;
        Address end; // exclusive
        FragmentId fragment_id;
        Offset fragment_offset;
        Flags local_flags;
        MappingOrigin origin;

        LiveSpan()
            : start(0), end(0), fragment_id(0), fragment_offset(0),
            local_flags(0), origin(MappingOrigin::WriteRoot) {}

        LiveSpan(Address s, Address e, FragmentId frag, Offset off, Flags local, MappingOrigin o)
            : start(s), end(e), fragment_id(frag), fragment_offset(off),
            local_flags(local), origin(o) {}
    };

    struct ResolvedLiveSpan {
        Address abs_start;
        Address abs_end;
        FragmentId fragment_id;
        Offset fragment_offset;
        Flags local_flags;

        ResolvedLiveSpan()
            : abs_start(0), abs_end(0), fragment_id(0), fragment_offset(0), local_flags(0) {}

        ResolvedLiveSpan(Address as, Address ae, FragmentId frag, Offset off, Flags local)
            : abs_start(as), abs_end(ae), fragment_id(frag), fragment_offset(off), local_flags(local) {}
    };

    // Current visible memory map: non-overlapping live spans keyed by start address.
    std::map<Address, LiveSpan> live_;

    // Fragment storage.
    std::map<FragmentId, Fragment> fragments_;

    // History log.
    std::vector<HistoryEntry> history_;

    FragmentId next_fragment_id_;

    static void appendMerged(std::vector<DetailedFlagSpan>& out, const DetailedFlagSpan& span) {
        if (span.start >= span.end) {
            return;
        }

        if (!out.empty() &&
            out.back().end == span.start &&
            out.back().flags == span.flags &&
            out.back().fragment_id == span.fragment_id &&
            out.back().origin == span.origin) {
            out.back().end = span.end;
        } else {
            out.push_back(span);
        }
    }

    void ensureRangeExists(Address start, Address end) {
        if (start >= end) {
            return;
        }

        Address pos = start;

        auto it = live_.upper_bound(start);
        if (it != live_.begin()) {
            --it;
        }

        while (pos < end) {
            while (it != live_.end() && it->second.end <= pos) {
                ++it;
            }

            if (it == live_.end() || it->second.start > pos) {
                const Address gapEnd =
                    (it == live_.end()) ? end : std::min(end, it->second.start);

#if ENABLE_HISTORY
                history_.emplace_back(OpKind::Write, pos, gapEnd - pos, 0, 0);
#endif
                const FragmentId frag = createFragment(gapEnd - pos, pos);
                insertLiveSpan(LiveSpan(pos, gapEnd, frag, 0, 0, MappingOrigin::WriteRoot));

                pos = gapEnd;
                continue;
            }

            pos = std::min(end, it->second.end);
        }

        coalesceLive();
    }

    static Address checkedEnd(Address addr, Size size) {
        if (size > 0 && addr > std::numeric_limits<Address>::max() - size) {
            throw std::overflow_error("address range overflow");
        }
        return addr + size;
    }

    FragmentId createFragment(Size size, Address source_write_address=0) {
        const FragmentId id = next_fragment_id_++;
        fragments_.emplace(id, Fragment(id, size, source_write_address));
        return id;
    }

    Fragment& getFragment(FragmentId id) {
        auto it = fragments_.find(id);
        if (it == fragments_.end()) {
            throw std::logic_error("missing fragment");
        }
        return it->second;
    }

    const Fragment& getFragmentConst(FragmentId id) const {
        auto it = fragments_.find(id);
        if (it == fragments_.end()) {
            throw std::logic_error("missing fragment");
        }
        return it->second;
    }

    void splitLiveAt(Address addr) {
        auto it = live_.upper_bound(addr);
        if (it == live_.begin()) {
            return;
        }

        --it;
        const LiveSpan span = it->second;

        if (addr <= span.start || addr >= span.end) {
            return;
        }

        const Offset delta = addr - span.start;

        it->second.end = addr;
        live_.emplace(addr, LiveSpan(
            addr,
            span.end,
            span.fragment_id,
            span.fragment_offset + delta,
            span.local_flags,
            span.origin));
    }

    void eraseLiveRange(Address start, Address end) {
        splitLiveAt(start);
        splitLiveAt(end);

        auto it = live_.lower_bound(start);
        while (it != live_.end() && it->second.start < end) {
            it = live_.erase(it);
        }
    }

    void insertLiveSpan(const LiveSpan& span) {
        if (span.start >= span.end) {
            return;
        }
        live_[span.start] = span;
    }

    void coalesceLive() {
        if (live_.empty()) {
            return;
        }

        auto it = live_.begin();
        while (it != live_.end()) {
            auto next = std::next(it);
            if (next == live_.end()) {
                break;
            }

            const LiveSpan& a = it->second;
            const LiveSpan& b = next->second;

            const bool sameFragment = (a.fragment_id == b.fragment_id);
            const bool contiguousAddr = (a.end == b.start);
            const bool contiguousFragment =
                (a.fragment_offset + (a.end - a.start) == b.fragment_offset);
            const bool sameLocalFlags = (a.local_flags == b.local_flags);

            const bool sameOrigin = (a.origin == b.origin);

            if (sameFragment && contiguousAddr && contiguousFragment && sameLocalFlags && sameOrigin) {
                it->second.end = b.end;
                live_.erase(next);
            } else {
                ++it;
            }
        }
    }

    // Resolve visible live mappings over [addr, addr+size), preserving:
    // - fragment identity
    // - fragment offset
    // - local flags
    //
    // This is used by copy() and markLineage().
    std::vector<ResolvedLiveSpan> resolveLiveMappings(Address addr, Size size) const {
        std::vector<ResolvedLiveSpan> out;
        if (size == 0) {
            return out;
        }

        const Address end = checkedEnd(addr, size);

        auto it = live_.upper_bound(addr);
        if (it != live_.begin()) {
            --it;
        }

        while (it != live_.end()) {
            const LiveSpan& live = it->second;

            if (live.end <= addr) {
                ++it;
                continue;
            }
            if (live.start >= end) {
                break;
            }

            const Address clipStart = std::max(addr, live.start);
            const Address clipEnd = std::min(end, live.end);
            const Offset fragOff = live.fragment_offset + (clipStart - live.start);

            appendMerged(out, ResolvedLiveSpan(
                clipStart,
                clipEnd,
                live.fragment_id,
                fragOff,
                live.local_flags));

            ++it;
        }

        return out;
    }

    template <typename Predicate>
    std::vector<Range> queryAllRangesByPredicate(Predicate pred) const {
        std::vector<Range> out;

        for (const auto& kv : live_) {
            const LiveSpan& live = kv.second;
            const Fragment& frag = getFragmentConst(live.fragment_id);

            const Offset fragStart = live.fragment_offset;
            const Offset fragEnd = live.fragment_offset + (live.end - live.start);

            const auto pieces = frag.queryRange(fragStart, fragEnd);
            for (const auto& piece : pieces) {
                const Address absStart = live.start + (piece.start - live.fragment_offset);
                const Address absEnd = live.start + (piece.end - live.fragment_offset);
                const Flags effective = live.local_flags | piece.flags;

                if (!pred(effective)) {
                    continue;
                }

                appendMerged(out, Range(absStart, absEnd));
            }
        }

        return out;
    }

    static void appendMerged(std::vector<FlagSpan>& out, const FlagSpan& span) {
        if (span.start >= span.end) {
            return;
        }

        if (!out.empty() &&
            out.back().end == span.start &&
            out.back().flags == span.flags) {
            out.back().end = span.end;
        } else {
            out.push_back(span);
        }
    }

    static void appendMerged(std::vector<Range>& out, const Range& range) {
        if (range.start >= range.end) {
            return;
        }

        if (!out.empty() && out.back().end == range.start) {
            out.back().end = range.end;
        } else {
            out.push_back(range);
        }
    }

    static void appendMerged(std::vector<ResolvedLiveSpan>& out, const ResolvedLiveSpan& span) {
        if (span.abs_start >= span.abs_end) {
            return;
        }

        if (!out.empty()) {
            ResolvedLiveSpan& last = out.back();
            const bool contiguousAddr = (last.abs_end == span.abs_start);
            const bool sameFragment = (last.fragment_id == span.fragment_id);
            const bool contiguousFragment =
                (last.fragment_offset + (last.abs_end - last.abs_start) == span.fragment_offset);
            const bool sameLocalFlags = (last.local_flags == span.local_flags);

            if (contiguousAddr && sameFragment && contiguousFragment && sameLocalFlags) {
                last.abs_end = span.abs_end;
                return;
            }
        }

        out.push_back(span);
    }
};

}  // namespace SPIRVSimulator

#endif
