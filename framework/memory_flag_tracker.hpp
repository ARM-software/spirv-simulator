#pragma once

#ifndef ARM_SPIRV_SIMULATOR_MEMORY_FLAG_TRACKER_HPP
#define ARM_SPIRV_SIMULATOR_MEMORY_FLAG_TRACKER_HPP

#include "checked_arithmetic.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
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
    using UniformDerivedClassId = std::uint64_t;

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
        MarkLineage,
        MarkUniformDerivedClass
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


    enum class MemoryPatternKind {
        Unknown,
        UniformDerivedElements
    };

    struct DetailedFlagSpan {
        Address start;
        Address end; // exclusive
        Address root_address;
        Flags flags;
        FragmentId fragment_id;
        MappingOrigin origin;

        DetailedFlagSpan()
            : start(0), end(0), root_address(0), flags(0), fragment_id(0),
            origin(MappingOrigin::WriteRoot) {}

        DetailedFlagSpan(Address s, Address e, Address r, Flags f, FragmentId frag, MappingOrigin o)
            : start(s), end(e), root_address(r), flags(f), fragment_id(frag), origin(o) {}
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
        coalesceLiveRange(addr, end);
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

        coalesceLiveRange(dst, dstEnd);
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

        coalesceLiveRange(addr, end);
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

        return QueryResult(addr, frag.source_write_address() + off, effective, live.fragment_id, live.origin);
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
                    frag.source_write_address() + sp.start,
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
    // Dormant persistent classification support for MemoryFlagTracker's broader memory type/taint tracking role.
    //
    // SPIRVSimulator's current physical-address path intentionally uses ElementwiseSource directly and
    // does not publish, query, or promote these classes.
    // The implementation is retained for future cross-memory and cross-dispatch payload classification.
    //
    // Before reconnecting it, callers must provide complete publication, external-write invalidation,
    // backing-allocation lifetime and generation tracking.

    // Ordered affine view over uniform-derived class ordinals. This type contains
    // only ordinal-set geometry.
    class ClassOrdinalSpan {
    public:
        enum class MergeResult {
            NotMergeable,
            Merged,
            Invalid,
        };

        Size first_class_ordinal = 0;
        int64_t class_ordinal_stride = 0;
        Size element_count = 0;

        bool operator==(const ClassOrdinalSpan&) const = default;

        // Returns whether every ordinal in other belongs to this span.
        bool contains(const ClassOrdinalSpan& other) const {
            ClassOrdinalSpan containing;
            ClassOrdinalSpan contained;
            if (!canonicalize(containing) ||
                !other.canonicalize(contained) ||
                containing.element_count < contained.element_count ||
                contained.first_class_ordinal < containing.first_class_ordinal) {
                return false;
            }

            const Size containing_stride = static_cast<Size>(containing.class_ordinal_stride);
            const Size delta = contained.first_class_ordinal - containing.first_class_ordinal;
            if ((delta % containing_stride) != 0) {
                return false;
            }

            const Size first_index = delta / containing_stride;
            if (first_index >= containing.element_count) {
                return false;
            }

            if (contained.element_count == 1) {
                return true;
            }

            const Size contained_stride = static_cast<Size>(contained.class_ordinal_stride);
            if ((contained_stride % containing_stride) != 0) {
                return false;
            }

            const Size index_stride = contained_stride / containing_stride;
            return contained.element_count - 1 <= (containing.element_count - 1 - first_index) / index_stride;
        }

        // Returns whether this span is in the unique canonical representation.
        bool isCanonical() const {
            if (element_count == 0 || class_ordinal_stride <= 0) {
                return false;
            }

            if (element_count == 1) {
                return class_ordinal_stride == 1;
            }

            const Size stride = static_cast<Size>(class_ordinal_stride);
            const Size step_count = element_count - 1;
            return step_count <= (std::numeric_limits<Size>::max() - first_class_ordinal) / stride;
        }

        // Converts this span to the canonical representation. On false, the out parameter is unchanged.
        bool canonicalize(ClassOrdinalSpan& canonical_span) const {
            if (element_count == 0) {
                return false;
            }

            ClassOrdinalSpan candidate = *this;
            if (element_count == 1) {
                candidate.class_ordinal_stride = 1;
                canonical_span = candidate;
                return true;
            }

            if (class_ordinal_stride == 0) {
                return false;
            }

            const Size step_count = element_count - 1;
            if (class_ordinal_stride > 0) {
                const Size stride = static_cast<Size>(class_ordinal_stride);
                if (step_count > (std::numeric_limits<Size>::max() - first_class_ordinal) / stride) {
                    return false;
                }
            } else {
                const Size stride_magnitude = Size{0} - static_cast<Size>(class_ordinal_stride);
                if (stride_magnitude > static_cast<Size>(std::numeric_limits<int64_t>::max()) ||
                    step_count > first_class_ordinal / stride_magnitude) {
                    return false;
                }

                const Size offset = step_count * stride_magnitude;
                candidate.first_class_ordinal = first_class_ordinal - offset;
                candidate.class_ordinal_stride = static_cast<int64_t>(stride_magnitude);
            }

            canonical_span = candidate;
            return true;
        }

        // Defines the stable lexicographical order used for canonical storage.
        // Both spans must be canonical.
        bool lessByCanonicalGeometry(const ClassOrdinalSpan& other) const {
            if (first_class_ordinal != other.first_class_ordinal) {
                return first_class_ordinal < other.first_class_ordinal;
            }

            if (class_ordinal_stride != other.class_ordinal_stride) {
                return class_ordinal_stride < other.class_ordinal_stride;
            }

            return element_count < other.element_count;
        }

        // Attempts to represent the exact union of two canonical,
        // non-containing spans by one canonical span.
        MergeResult tryMergeCanonicalUnion(const ClassOrdinalSpan& other, ClassOrdinalSpan& merged) const {
            if (!isCanonical() || !other.isCanonical()) {
                return MergeResult::Invalid;
            }

            if (element_count == 1 && other.element_count == 1) {
                return tryMergeTwoSingletons(other, merged);
            }

            if (element_count == 1) {
                return other.tryExtendWithSingleton(*this, merged);
            }

            if (other.element_count == 1) {
                return tryExtendWithSingleton(other, merged);
            }

            return tryMergeSameLattice(other, merged);
        }

    private:
        // Both operands must be canonical singletons. Equality is normally removed by the caller's containment check.
        MergeResult tryMergeTwoSingletons(const ClassOrdinalSpan& singleton, ClassOrdinalSpan& merged) const {
            const Size first = std::min(first_class_ordinal, singleton.first_class_ordinal);
            const Size last = std::max(first_class_ordinal, singleton.first_class_ordinal);
            if (last - first != 1) {
                return MergeResult::NotMergeable;
            }

            ClassOrdinalSpan candidate;
            candidate.first_class_ordinal = first;
            candidate.class_ordinal_stride = 1;
            candidate.element_count = 2;
            merged = candidate;
            return MergeResult::Merged;
        }

        // This span must be canonical and multi-element. Containment is handled by the caller.
        MergeResult tryExtendWithSingleton(const ClassOrdinalSpan& singleton, ClassOrdinalSpan& merged) const {
            const Size stride = static_cast<Size>(class_ordinal_stride);
            const Size last = first_class_ordinal + (element_count - 1) * stride;
            const Size singleton_ordinal = singleton.first_class_ordinal;

            Size merged_first = first_class_ordinal;
            if (singleton_ordinal < first_class_ordinal) {
                if (first_class_ordinal - singleton_ordinal != stride) {
                    return MergeResult::NotMergeable;
                }

                merged_first = singleton_ordinal;
            } else {
                if (singleton_ordinal <= last) {
                    return MergeResult::NotMergeable;
                }

                if (singleton_ordinal - last != stride) {
                    return MergeResult::NotMergeable;
                }
            }

            if (element_count == std::numeric_limits<Size>::max()) {
                return MergeResult::Invalid;
            }

            ClassOrdinalSpan candidate;
            candidate.first_class_ordinal = merged_first;
            candidate.class_ordinal_stride = class_ordinal_stride;
            candidate.element_count = element_count + 1;
            merged = candidate;
            return MergeResult::Merged;
        }

        // Both operands must be canonical multi-element spans. Containment is
        // handled by the caller; this function checks same-lattice compatibility.
        MergeResult tryMergeSameLattice(const ClassOrdinalSpan& other, ClassOrdinalSpan& merged) const {
            if (class_ordinal_stride != other.class_ordinal_stride) {
                return MergeResult::NotMergeable;
            }

            const ClassOrdinalSpan& left = first_class_ordinal <= other.first_class_ordinal ? *this : other;
            const ClassOrdinalSpan& right = first_class_ordinal <= other.first_class_ordinal ? other : *this;

            const Size stride = static_cast<Size>(left.class_ordinal_stride);
            const Size first_delta = right.first_class_ordinal - left.first_class_ordinal;
            if ((first_delta % stride) != 0) {
                return MergeResult::NotMergeable;
            }

            const Size left_last = left.first_class_ordinal + (left.element_count - 1) * stride;
            const Size right_last = right.first_class_ordinal + (right.element_count - 1) * stride;

            if (right.first_class_ordinal > left_last && right.first_class_ordinal - left_last != stride) {
                return MergeResult::NotMergeable;
            }

            const Size merged_last = std::max(left_last, right_last);
            const Size merged_step_count = (merged_last - left.first_class_ordinal) / stride;
            if (merged_step_count == std::numeric_limits<Size>::max()) {
                return MergeResult::Invalid;
            }

            ClassOrdinalSpan candidate;
            candidate.first_class_ordinal = left.first_class_ordinal;
            candidate.class_ordinal_stride = left.class_ordinal_stride;
            candidate.element_count = merged_step_count + 1;
            merged = candidate;
            return MergeResult::Merged;
        }
    };

    struct PromotedOrdinalView {
        ClassOrdinalSpan ordinals;
        Flags flags = 0;
    };

    // Persistent geometry and promotion state for one uniform-derived class.
    class UniformDerivedClass {
    public:
        UniformDerivedClassId class_id = 0;

        // Immutable destination geometry copied from the publishing span.
        Address first_address = 0;
        int64_t byte_stride = 0;
        Size element_count = 0;
        Size element_size = 0;

        // Properties proved for every element when the class is published.
        Flags class_flags = 0;
        MemoryPatternKind kind = MemoryPatternKind::Unknown;

        // Properties proved after publication.
        Flags whole_class_promoted_flags = 0;
        std::vector<PromotedOrdinalView> partial_promotions;

        // Existing views may retain an inactive record after registry removal.
        bool active = true;

        // Returns the ordered class-ordinal mapping when the complete load
        // geometry is an affine view of this active class.
        std::optional<ClassOrdinalSpan> queryOrdinalSpan(Address load_first_address,
                                                         int64_t load_byte_stride,
                                                         Size load_element_count,
                                                         Size load_element_size) const {
            if (!active ||
                kind != MemoryPatternKind::UniformDerivedElements ||
                load_element_count == 0 ||
                load_element_size == 0 ||
                load_element_size != element_size ||
                load_element_count > element_count) {
                return std::nullopt;
            }

            Size first_ordinal = 0;
            if (!tryGetExactOrdinal(load_first_address, first_ordinal)) {
                return std::nullopt;
            }

            int64_t ordinal_stride = 1;
            if (load_element_count > 1 &&
                (!CheckedExactDivideI64ByI64(load_byte_stride, byte_stride, ordinal_stride) ||
                 ordinal_stride == 0)) {
                return std::nullopt;
            }

            const ClassOrdinalSpan ordinal_span{
                first_ordinal,
                ordinal_stride,
                load_element_count,
            };

            // Preserve the raw load order in the result; canonicalization is
            // used only for endpoint and owning-class bounds validation.
            ClassOrdinalSpan canonical_span;
            if (!ordinal_span.canonicalize(canonical_span) ||
                !containsOrdinalSpan(canonical_span)) {
                return std::nullopt;
            }

            return ordinal_span;
        }

        // Promotes flags for the supplied ordinal view. Recoverable validation
        // failure leaves all persistent state unchanged.
        bool promoteOrdinalSpan(const ClassOrdinalSpan& span, Flags flags) {
            if (span.element_count == 0 || flags == 0) {
                return false;
            }

            ClassOrdinalSpan canonical_span;
            if (!span.canonicalize(canonical_span) ||
                !containsOrdinalSpan(canonical_span)) {
                return false;
            }

            // A retained SSA view remains valid, but an inactive record no
            // longer denotes current memory and must not be mutated.
            if (!active) {
                return true;
            }

            if (coversWholeClass(canonical_span)) {
                whole_class_promoted_flags |= flags;
                removePartialPromotionFlags(partial_promotions, flags);
                return true;
            }

            flags &= ~whole_class_promoted_flags;
            if (flags == 0) {
                return true;
            }

            return mergePartialPromotion(canonical_span, flags);
        }

        // Recovers the exact class ordinal of one element start.
        bool tryGetExactOrdinal(Address element_start, Size& ordinal) const {
            if (element_count == 0) {
                return false;
            }

            if (element_count == 1) {
                if (element_start != first_address) {
                    return false;
                }

                ordinal = 0;
                return true;
            }

            if (byte_stride == 0) {
                return false;
            }

            Address delta = 0;
            Size step = 0;
            if (byte_stride > 0) {
                if (element_start < first_address) {
                    return false;
                }

                delta = element_start - first_address;
                step = static_cast<Size>(byte_stride);
            } else {
                if (element_start > first_address) {
                    return false;
                }

                delta = first_address - element_start;
                step = MagnitudeI64(byte_stride);
            }

            if ((delta % step) != 0) {
                return false;
            }

            const Size candidate_ordinal = delta / step;
            if (candidate_ordinal >= element_count) {
                return false;
            }

            ordinal = candidate_ordinal;
            return true;
        }

        // Returns whether a canonical span lies entirely within this class's ordinal bounds.
        bool containsOrdinalSpan(const ClassOrdinalSpan& canonical_span) const {
            if (element_count == 0 ||
                !canonical_span.isCanonical() ||
                canonical_span.first_class_ordinal >= element_count) {
                return false;
            }

            const Size step_count = canonical_span.element_count - 1;
            const Size stride = static_cast<Size>(canonical_span.class_ordinal_stride);
            return step_count <= (element_count - 1 - canonical_span.first_class_ordinal) / stride;
        }

        // Returns whether span denotes exactly every class ordinal. The input
        // may be either the forward or reverse full view.
        bool coversWholeClass(const ClassOrdinalSpan& span) const {
            if (span.element_count == 0 || span.element_count != element_count) {
                return false;
            }

            if (element_count == 1) {
                return span.first_class_ordinal == 0;
            }

            return (span.first_class_ordinal == 0 && span.class_ordinal_stride == 1) ||
                   (span.first_class_ordinal == element_count - 1 &&
                    span.class_ordinal_stride == -1);
        }

    private:
        // Adds flags for canonical_span to partial_promotions and may upgrade
        // them to whole_class_promoted_flags.
        bool mergePartialPromotion(const ClassOrdinalSpan& canonical_span, Flags flags) {
            Flags candidate_whole_flags = whole_class_promoted_flags;
            std::vector<PromotedOrdinalView> candidate = partial_promotions;
            std::vector<PromotedOrdinalView> normalized_entries;

            for (Flags remaining = flags; remaining != 0; remaining &= remaining - 1) {
                const Flags bit = remaining & (Flags{0} - remaining);
                std::vector<ClassOrdinalSpan> spans;

                for (PromotedOrdinalView& promotion : candidate) {
                    if ((promotion.flags & bit) != 0) {
                        spans.push_back(promotion.ordinals);
                        promotion.flags &= ~bit;
                    }
                }

                if (!insertAndNormalizeOrdinalSpan(spans, canonical_span)) {
                    return false;
                }

                if (spans.size() == 1 && coversWholeClass(spans.front())) {
                    candidate_whole_flags |= bit;
                } else {
                    for (const ClassOrdinalSpan& normalized_span : spans) {
                        normalized_entries.push_back({ normalized_span, bit });
                    }
                }
            }

            candidate.insert(candidate.end(), normalized_entries.begin(), normalized_entries.end());

            removePartialPromotionFlags(candidate, candidate_whole_flags);
            sortPartialPromotionsByOrdinalSpan(candidate);
            combineFlagsForIdenticalSpans(candidate);

            partial_promotions.swap(candidate);
            whole_class_promoted_flags = candidate_whole_flags;
            return true;
        }

        static void removePartialPromotionFlags(std::vector<PromotedOrdinalView>& promotions, Flags flags) {
            for (PromotedOrdinalView& promotion : promotions) {
                promotion.flags &= ~flags;
            }

            promotions.erase(
                std::remove_if(
                    promotions.begin(),
                    promotions.end(),
                    [](const PromotedOrdinalView& promotion) {
                        return promotion.flags == 0;
                    }),
                promotions.end());
        }

        static void sortPartialPromotionsByOrdinalSpan(std::vector<PromotedOrdinalView>& promotions) {
            std::sort(
                promotions.begin(),
                promotions.end(),
                [](const PromotedOrdinalView& lhs, const PromotedOrdinalView& rhs) {
                    return lhs.ordinals.lessByCanonicalGeometry(rhs.ordinals);
                });
        }

        static void combineFlagsForIdenticalSpans(std::vector<PromotedOrdinalView>& promotions) {
            if (promotions.empty()) {
                return;
            }

            auto output = promotions.begin();
            for (auto input = std::next(promotions.begin()); input != promotions.end(); ++input) {
                if (output->ordinals == input->ordinals) {
                    output->flags |= input->flags;
                    continue;
                }

                ++output;
                if (output != input) {
                    *output = std::move(*input);
                }
            }

            promotions.erase(std::next(output), promotions.end());
        }

        // Normalizes canonical spans belonging to one flag bit. On a
        // recoverable false return, spans is unchanged.
        static bool normalizeOrdinalSpans(std::vector<ClassOrdinalSpan>& spans) {
            std::vector<ClassOrdinalSpan> normalized;
            normalized.reserve(spans.size());

            for (const ClassOrdinalSpan& span : spans) {
                if (!span.isCanonical() ||
                    !insertAndNormalizeOrdinalSpan(normalized, span)) {
                    return false;
                }
            }

            spans.swap(normalized);
            return true;
        }

        // Inserts canonical_span into an already normalized span set. On a
        // recoverable false return, normalized is unchanged.
        static bool insertAndNormalizeOrdinalSpan(
            std::vector<ClassOrdinalSpan>& normalized,
            const ClassOrdinalSpan& canonical_span) {
            if (!canonical_span.isCanonical()) {
                return false;
            }

            ClassOrdinalSpan candidate = canonical_span;
            std::vector<uint8_t> removed(normalized.size(), false);
            bool candidate_absorbed = false;

            for (size_t index = 0; index < normalized.size();) {
                if (removed[index]) {
                    ++index;
                    continue;
                }

                const ClassOrdinalSpan& existing = normalized[index];
                if (existing.contains(candidate)) {
                    candidate_absorbed = true;
                    break;
                }

                if (candidate.contains(existing)) {
                    removed[index] = true;
                    ++index;
                    continue;
                }

                ClassOrdinalSpan merged;
                const ClassOrdinalSpan::MergeResult result =
                    existing.tryMergeCanonicalUnion(candidate, merged);
                if (result == ClassOrdinalSpan::MergeResult::Invalid) {
                    return false;
                }

                if (result == ClassOrdinalSpan::MergeResult::Merged) {
                    candidate = merged;
                    removed[index] = true;
                    index = 0;
                    continue;
                }

                ++index;
            }

            std::vector<ClassOrdinalSpan> committed;
            committed.reserve(normalized.size() + (candidate_absorbed ? 0 : 1));
            for (size_t index = 0; index < normalized.size(); ++index) {
                if (!removed[index]) {
                    committed.push_back(normalized[index]);
                }
            }

            if (!candidate_absorbed) {
                committed.push_back(candidate);
            }

            sortCanonicalSpans(committed);
            normalized.swap(committed);
            return true;
        }

        static void sortCanonicalSpans(std::vector<ClassOrdinalSpan>& spans) {
            std::sort(
                spans.begin(),
                spans.end(),
                [](const ClassOrdinalSpan& lhs, const ClassOrdinalSpan& rhs) {
                    return lhs.lessByCanonicalGeometry(rhs);
                });
        }

        static bool isNormalizedOrdinalSpanSet(
            const std::vector<ClassOrdinalSpan>& spans) {
            for (size_t index = 0; index < spans.size(); ++index) {
                const ClassOrdinalSpan& span = spans[index];
                if (!span.isCanonical()) {
                    return false;
                }

                if (index != 0 && span.lessByCanonicalGeometry(spans[index - 1])) {
                    return false;
                }

                for (size_t previous_index = 0; previous_index < index; ++previous_index) {
                    const ClassOrdinalSpan& previous = spans[previous_index];
                    if (previous.contains(span) || span.contains(previous)) {
                        return false;
                    }

                    ClassOrdinalSpan merged;
                    const ClassOrdinalSpan::MergeResult result = previous.tryMergeCanonicalUnion(span, merged);
                    if (result != ClassOrdinalSpan::MergeResult::NotMergeable) {
                        return false;
                    }
                }
            }

            return true;
        }
    };

public:
    bool hasActiveUniformDerivedClasses() const {
        return !active_uniform_derived_classes_.empty();
    }

    // Retire every active class whose actual element bytes overlap the
    // caller-provided destination geometry. Existing views keep the record
    // alive, but registry queries can no longer match it.
    template <typename OverlapPredicate>
    void invalidateUniformDerivedClassesIf(OverlapPredicate&& overlaps) {
        std::erase_if(
            active_uniform_derived_classes_,
            [&](const std::shared_ptr<UniformDerivedClass>& candidate) {
                if (!candidate || !candidate->active) {
                    return true;
                }

                if (!overlaps(candidate->first_address,
                              candidate->byte_stride,
                              candidate->element_count,
                              candidate->element_size)) {
                    return false;
                }

                candidate->active = false;
                return true;
            });
    }

    // Stable, opaque load-time relation between an SSA value and a
    // uniform-derived class's ordered ordinal set.
    class UniformDerivedClassView {
    public:
        UniformDerivedClassView() = default;

        bool valid() const {
            return class_record_ != nullptr && ordinals_.element_count != 0;
        }

        UniformDerivedClassId classId() const {
            return valid() ? class_record_->class_id : 0;
        }

        Size elementCount() const {
            return valid() ? ordinals_.element_count : 0;
        }

        Size elementSize() const {
            return valid() ? class_record_->element_size : 0;
        }

        bool operator==(const UniformDerivedClassView&) const = default;

    private:
        friend class MemoryFlagTracker;

        UniformDerivedClassView(std::shared_ptr<UniformDerivedClass> class_record, ClassOrdinalSpan ordinals)
            : class_record_(std::move(class_record)),
              ordinals_(std::move(ordinals)) {}

        // Returns flags that hold for every ordinal in this view. Existing
        // views remain queryable after their class record becomes inactive.
        Flags query() const {
            if (!valid()) {
                return 0;
            }

            return class_record_->class_flags |
                   class_record_->whole_class_promoted_flags |
                   queryPartialPromotionFlags();
        }

        // Promotes flags for this complete view into its persistent class
        // record, including the class's active/inactive semantics.
        bool promote(Flags flags) const {
            if (!valid()) {
                return false;
            }
            return class_record_->promoteOrdinalSpan(ordinals_, flags);
        }

        Flags queryPartialPromotionFlags() const {
            Flags flags = 0;
            for (const PromotedOrdinalView& promotion : class_record_->partial_promotions) {
                if (promotion.flags != 0 && promotion.ordinals.contains(ordinals_)) {
                    flags |= promotion.flags;
                }
            }
            return flags;
        }

        std::shared_ptr<UniformDerivedClass> class_record_;
        ClassOrdinalSpan ordinals_;
    };

    // Return the unique active class whose complete element geometry contains
    // the requested ordered affine load view.
    std::optional<UniformDerivedClassView> queryUniformDerivedClassView(
        Address load_first_address,
        int64_t load_byte_stride,
        Size load_element_count,
        Size load_element_size) const
    {
        std::optional<UniformDerivedClassView> match;

        for (const std::shared_ptr<UniformDerivedClass>& candidate : active_uniform_derived_classes_)
        {
            if (!candidate)
            {
                continue;
            }

            const std::optional<ClassOrdinalSpan> ordinals = candidate->queryOrdinalSpan(
                load_first_address, load_byte_stride, load_element_count, load_element_size);
            if (!ordinals)
            {
                continue;
            }

            if (match)
            {
                // Ambiguous active classification must not depend on registry order.
                return std::nullopt;
            }

            match = UniformDerivedClassView(candidate, *ordinals);
        }

        return match;
    }

    // Recover flags that hold for every ordinal represented by view.
    Flags query(const UniformDerivedClassView& view) const
    {
        return view.query();
    }

    // Promote flags for every class ordinal represented by view.
    bool promoteUniformDerivedClassView(const UniformDerivedClassView& view, Flags flags)
    {
        return view.promote(flags);
    }

    // Register a persistent strided class after the caller has proved and
    // committed the complete destination element sequence.
    bool markUniformDerivedClass(Address first_address,
                                 int64_t byte_stride,
                                 Size element_count,
                                 Size element_size,
                                 Flags flags)
    {
        if (first_address == 0 || element_count == 0 || element_size == 0 ||
            (element_count > 1 && byte_stride == 0))
        {
            return false;
        }

        if (next_uniform_derived_class_id_ == 0 ||
            next_uniform_derived_class_id_ == std::numeric_limits<UniformDerivedClassId>::max())
        {
            return false;
        }

        const UniformDerivedClassId class_id = next_uniform_derived_class_id_;
        std::shared_ptr<UniformDerivedClass> candidate = std::make_shared<UniformDerivedClass>();
        candidate->class_id = class_id;
        candidate->first_address = first_address;
        candidate->byte_stride = byte_stride;
        candidate->element_count = element_count;
        candidate->element_size = element_size;
        candidate->class_flags = flags;
        candidate->kind = MemoryPatternKind::UniformDerivedElements;
        candidate->whole_class_promoted_flags = 0;
        candidate->active = true;

        // Commit only after the candidate is fully initialized.
        active_uniform_derived_classes_.push_back(candidate);
        ++next_uniform_derived_class_id_;

#if ENABLE_HISTORY
        history_.emplace_back(
            OpKind::MarkUniformDerivedClass,
            first_address,
            element_count,
            flags,
            class_id);
#endif

        return true;
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
    std::vector<std::shared_ptr<UniformDerivedClass>> active_uniform_derived_classes_;

    FragmentId next_fragment_id_;
    UniformDerivedClassId next_uniform_derived_class_id_ = 1;

    static void appendMerged(std::vector<DetailedFlagSpan>& out, const DetailedFlagSpan& span) {
        if (span.start >= span.end) {
            return;
        }

        if (!out.empty() &&
            out.back().end == span.start &&
            out.back().root_address + (out.back().end - out.back().start) == span.root_address &&
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

        coalesceLiveRange(start, end);
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

    void coalesceLiveRange(Address start, Address end) {
        if (live_.empty()) {
            return;
        }

        auto it = live_.lower_bound(start);
        if (it != live_.begin()) {
            --it;
        }

        while (it != live_.end()) {
            auto next = std::next(it);
            if (next == live_.end()) {
                break;
            }

            const LiveSpan& a = it->second;
            const LiveSpan& b = next->second;

            if (a.start > end && b.start > end) {
                break;
            }

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
