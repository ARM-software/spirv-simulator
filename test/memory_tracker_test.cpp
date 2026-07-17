#include <gtest/gtest.h>
#include "../framework/memory_flag_tracker.hpp"

using SPIRVSimulator::MemoryFlagTracker;

class MemoryFlagTrackerTest : public ::testing::Test {
protected:
    MemoryFlagTracker tracker;
    uint64_t FLAG_A = 1ull << 0;
    uint64_t FLAG_B = 1ull << 1;
    uint64_t FLAG_C = 1ull << 2;

    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(MemoryFlagTrackerTest, WriteAndQuery) {

    // Write flags to a range
    tracker.write(100, 10, FLAG_A);

    // Query inside the range
    auto q1 = tracker.query(100);
    ASSERT_TRUE(q1.has_value());
    EXPECT_EQ(q1.value(), FLAG_A);

    auto q2 = tracker.query(105);
    ASSERT_TRUE(q2.has_value());
    EXPECT_EQ(q2.value(), FLAG_A);

    auto q3 = tracker.query(109);
    ASSERT_TRUE(q3.has_value());
    EXPECT_EQ(q3.value(), FLAG_A);

    // Query outside the range
    auto q4 = tracker.query(99);
    EXPECT_FALSE(q4.has_value());

    auto q5 = tracker.query(110);
    EXPECT_FALSE(q5.has_value());

    // Overwrite partially
    tracker.write(105, 10, FLAG_B);

    EXPECT_EQ(tracker.query(104).value(), FLAG_A);
    EXPECT_EQ(tracker.query(105).value(), FLAG_B);
    EXPECT_EQ(tracker.query(114).value(), FLAG_B);
    EXPECT_FALSE(tracker.query(115).has_value());
}

TEST_F(MemoryFlagTrackerTest, MarkLocalAndLineage) {

    tracker.write(100, 10, FLAG_A);
    tracker.copy(100, 200, 10);
    tracker.copy(100, 300, 10);

    // markRange only affects the specified local span
    tracker.markRange(300, 5, FLAG_B);
    EXPECT_EQ(tracker.query(302).value(), FLAG_A | FLAG_B);
    EXPECT_EQ(tracker.query(102).value(), FLAG_A); // unchanged
    EXPECT_EQ(tracker.query(202).value(), FLAG_A); // unchanged

    // markLineage affects all spans sharing the fragment
    tracker.markLineage(205, 5, FLAG_C);

    // Affected lineage for offet 5-10
    EXPECT_EQ(tracker.query(106).value(), FLAG_A | FLAG_C);
    EXPECT_EQ(tracker.query(206).value(), FLAG_A | FLAG_C);
    EXPECT_EQ(tracker.query(306).value(), FLAG_A | FLAG_C);

    // Unaffected part
    EXPECT_EQ(tracker.query(102).value(), FLAG_A);
    EXPECT_EQ(tracker.query(202).value(), FLAG_A);
}

TEST_F(MemoryFlagTrackerTest, OverlapsAndGaps) {
    // A gap in the middle
    tracker.write(100, 5, FLAG_A);
    tracker.write(110, 5, FLAG_A);

    // Copying over a range covering mapped and unmapped
    tracker.copy(100, 200, 15);

    // [200, 205) should have FLAG_A
    EXPECT_EQ(tracker.query(202).value(), FLAG_A);
    // [205, 210) should be 0 because it was unmapped in source
    EXPECT_EQ(tracker.query(207).value(), 0);
    // [210, 215) should have FLAG_A
    EXPECT_EQ(tracker.query(212).value(), FLAG_A);

    // Copying with a destination that overlaps source
    tracker.write(300, 10, FLAG_B);
    tracker.copy(305, 300, 10); // Shift left by 5
    // source was [305, 310) FLAG_B, [310, 315) unmapped, but ensureRangeExists materializes [310, 315)
    EXPECT_EQ(tracker.query(300).value(), FLAG_B);
    EXPECT_EQ(tracker.query(305).value(), 0);

    // 310 to 315 was unmapped, but reading it as source materializes it with 0
    ASSERT_TRUE(tracker.query(310).has_value());
    EXPECT_EQ(tracker.query(310).value(), 0);
}

TEST_F(MemoryFlagTrackerTest, ZeroSizeOperations) {
    // Writing size 0 should do nothing
    tracker.write(100, 0, FLAG_A);
    EXPECT_FALSE(tracker.query(100).has_value());

    // Copying size 0 should do nothing
    tracker.write(100, 10, FLAG_A);
    tracker.copy(100, 200, 0);
    EXPECT_FALSE(tracker.query(200).has_value());

    // Marking size 0 should do nothing
    tracker.markRange(100, 0, FLAG_A);
    tracker.markLineage(100, 0, FLAG_A);
}

TEST_F(MemoryFlagTrackerTest, RangeQueries) {
    tracker.write(100, 10, FLAG_A);
    tracker.write(120, 10, FLAG_B);
    tracker.write(140, 10, FLAG_A | FLAG_B);

    // queryRange
    auto ranges = tracker.queryRange(90, 150);
    ASSERT_EQ(ranges.size(), 3);
    EXPECT_EQ(ranges[0].start, 100);
    EXPECT_EQ(ranges[0].end, 110);
    EXPECT_EQ(ranges[0].flags, FLAG_A);

    EXPECT_EQ(ranges[1].start, 120);
    EXPECT_EQ(ranges[1].end, 130);
    EXPECT_EQ(ranges[1].flags, FLAG_B);

    EXPECT_EQ(ranges[2].start, 140);
    EXPECT_EQ(ranges[2].end, 150);
    EXPECT_EQ(ranges[2].flags, FLAG_A | FLAG_B);

    // queryRangeDetailed
    // We wrote FLAG_A at 100..110, FLAG_B at 120..130, FLAG_A | FLAG_B at 140..150
    auto detailedRanges = tracker.queryRangeDetailed(115, 150); // Query across multiple
    ASSERT_EQ(detailedRanges.size(), 2);

    // It returns [120, 130) and [140, 150)
    EXPECT_EQ(detailedRanges[0].start, 120);
    EXPECT_EQ(detailedRanges[0].end, 130);
    EXPECT_EQ(detailedRanges[0].flags, FLAG_B);
    EXPECT_EQ(detailedRanges[0].root_address, 120);

    EXPECT_EQ(detailedRanges[1].start, 140);
    EXPECT_EQ(detailedRanges[1].end, 150);
    EXPECT_EQ(detailedRanges[1].flags, FLAG_A | FLAG_B);
    EXPECT_EQ(detailedRanges[1].root_address, 140);

    // queryAllRangesWithAnyFlags
    auto anyFlagsRanges = tracker.queryAllRangesWithAnyFlags(FLAG_A);
    ASSERT_EQ(anyFlagsRanges.size(), 2);
    EXPECT_EQ(anyFlagsRanges[0].start, 100);
    EXPECT_EQ(anyFlagsRanges[1].start, 140);

    // queryAllRangesWithAllFlags
    auto allFlagsRanges = tracker.queryAllRangesWithAllFlags(FLAG_A | FLAG_B);
    ASSERT_EQ(allFlagsRanges.size(), 1);
    EXPECT_EQ(allFlagsRanges[0].start, 140);
    EXPECT_EQ(allFlagsRanges[0].end, 150);
}

TEST_F(MemoryFlagTrackerTest, CopyAndQuery) {
    tracker.write(100, 10, FLAG_A);

    // Copy to another range
    tracker.copy(100, 200, 10);

    auto q1 = tracker.query(205);
    ASSERT_TRUE(q1.has_value());
    EXPECT_EQ(q1.value(), FLAG_A);

    // Querying detailed shows they share fragment lineage
    auto d1 = tracker.queryDetailed(105);
    auto d2 = tracker.queryDetailed(205);

    ASSERT_TRUE(d1.has_value());
    ASSERT_TRUE(d2.has_value());
    EXPECT_EQ(d1->fragment_id, d2->fragment_id);
    EXPECT_EQ(d1->flags, FLAG_A);
    EXPECT_EQ(d2->flags, FLAG_A);

    // Copy from unmapped memory initializes dest with 0
    tracker.copy(300, 400, 5);
    auto q_unmapped = tracker.query(402);
    ASSERT_TRUE(q_unmapped.has_value());
    EXPECT_EQ(q_unmapped.value(), 0);
}
