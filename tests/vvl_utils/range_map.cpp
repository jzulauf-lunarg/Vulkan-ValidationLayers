/*
 * Copyright (c) 2023 The Khronos Group Inc.
 * Copyright (c) 2023 Valve Corporation
 * Copyright (c) 2023 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <vector>
#include "../framework/test_common.h"
#include "containers/range_vector.h"

using TestMap = sparse_container::range_map<unsigned, int>;
using TestRange = TestMap::key_type;
using TestMapped = TestMap::mapped_type;
using TestValue = std::pair<TestRange, TestMapped>;
using TestVector = std::vector<TestValue>;
using TestMapGen = sparse_container::mapped_range_generator<TestMap>;

bool MapRefMatches(const TestMap& map, const TestVector& ref) {
    auto comp = ref.begin();
    bool diff = ref.size() != map.size();
    for (const auto& val : map) {
        diff |= (val.first != comp->first);
        ++comp;
    }
    return !diff;
};

void LoadMap(TestMap& map, const TestVector& ref) {
    for (const auto& val : ref) {
        map.insert(val);
    }
};

bool RangeSort(const TestValue& left, const TestValue& right) { return left.first.begin < right.first.begin; };

// Reference array
// Will delete 0xDEAD, and consolidate 0xF00D
static const size_t kDeadEntry = 2;
static const size_t kSameEntry0 = 3;
static const size_t kSameEntry1 = 4;
static const size_t kAnswerEntry = 0;
static const size_t kGapCheckEntry = 5;
static const TestMapped kInfillValue = -999;
static const TestRange::index_type kMinRangeIndex = std::numeric_limits<TestRange::index_type>::min();
static const TestRange::index_type kMaxRangeIndex = std::numeric_limits<TestRange::index_type>::max();

static const TestVector kTestInputVector{{{50, 52}, 42},     {{10, 21}, 2},      {{75, 100}, 0xDEAD},
                                         {{31, 45}, 0xF00D}, {{45, 50}, 0xF00D}, {{101, 108}, 8675309}};

TestVector SortedRef() {
    TestVector sorted = kTestInputVector;
    std::sort(sorted.begin(), sorted.end(), RangeSort);
    return sorted;
}

TEST(CustomContainer, RangeMapInsertDeleteMerge) {
    auto sorted = SortedRef();
    TestMap map0;

    // Test can we load the map in an unsorted way
    LoadMap(map0, kTestInputVector);
    ASSERT_TRUE(MapRefMatches(map0, sorted));

    // Check item deletion (and lower_bound)
    TestVector undead = kTestInputVector;
    ASSERT_TRUE(undead[kDeadEntry].second == 0xDEAD);
    const auto dead_range = undead[kDeadEntry].first;
    undead.erase(undead.begin() + kDeadEntry);
    std::sort(undead.begin(), undead.end(), RangeSort);
    auto dead = map0.lower_bound(dead_range);
    ASSERT_TRUE(dead != map0.end());

    if (dead != map0.end()) {
        map0.erase(dead);
        ASSERT_TRUE(MapRefMatches(map0, undead));
    }

    // Check item insertion
    map0.insert(kTestInputVector[kDeadEntry]);
    ASSERT_TRUE(MapRefMatches(map0, sorted));

    // Check consolidation
    auto consolidated = kTestInputVector;
    ASSERT_TRUE(consolidated[kSameEntry0].second == 0xF00D);
    ASSERT_TRUE(consolidated[kSameEntry1].second == 0xF00D);
    const TestRange same_range{consolidated[kSameEntry0].first.begin, consolidated[kSameEntry1].first.end};
    consolidated.erase(consolidated.begin() + kSameEntry1);
    consolidated[kSameEntry0].first = same_range;
    std::sort(consolidated.begin(), consolidated.end(), RangeSort);

    sparse_container::consolidate(map0);
    ASSERT_TRUE(MapRefMatches(map0, consolidated));
}

TEST(CustomContainer, RangeMapGenerators) {
    auto sorted = SortedRef();
    TestMap map0;
    LoadMap(map0, kTestInputVector);

    TestMapGen gen0(map0);

    auto comp_gen = [](TestMapGen& gen, const TestVector& ref) {
        bool diff = false;
        auto comp = ref.begin();
        for (; !gen->empty(); ++gen) {
            if (comp == ref.end()) {
                diff = true;
            } else {
                diff |= *gen != comp->first;
            }
            ++comp;
        }
        diff |= (comp != ref.end());
        return !diff;
    };
    ASSERT_TRUE(comp_gen(gen0, sorted));

    sparse_container::mapped_range_generator<TestMap> gen1(map0, kTestInputVector[kAnswerEntry].first);
    ASSERT_TRUE(*gen1 == kTestInputVector[kAnswerEntry].first);

    sparse_container::mapped_range_generator<TestMap> gen2(map0);
    gen2.seek_forward(kTestInputVector[kAnswerEntry].first);
    auto truncated = sorted;
    while (*truncated.begin() != kTestInputVector[kAnswerEntry]) {
        truncated.erase(truncated.begin());
    }
    ASSERT_TRUE(comp_gen(gen2, truncated));

    // Test infiling map0
    sparse_container::mapped_gap_generator<TestMap> gap0(map0);
    TestMap map1(map0);
    for (; !gap0->empty(); ++gap0) {
        map1.insert(TestValue(*gap0, kInfillValue));
    }

    // Verify the infill operation worked
    ASSERT_TRUE(map1.cbegin()->first.begin == kMinRangeIndex);
    auto last_in_map1 = map1.cend();
    --last_in_map1;
    ASSERT_TRUE(last_in_map1->first.end == kMaxRangeIndex);
    ASSERT_TRUE(map1.cbegin()->second == kInfillValue);
    TestRange first_range = map1.cbegin()->first;
    TestRange last_range = first_range;
    bool last_was_gap = false;
    for (const auto& value : map1) {
        bool is_gap = (value.second == kInfillValue);

        if (value.first != first_range) {
            ASSERT_TRUE(value.first.begin == last_range.end);
            ASSERT_TRUE(!(last_was_gap && is_gap));
            bool found_in_ref = map0.end() != map0.find(value.first);

            ASSERT_TRUE(found_in_ref != is_gap);
        }
        last_range = value.first;
        last_was_gap = is_gap;
    }

    // The "current" gap for a specified range is the gap that satifies the "lower_bound" criteria for relative to the gaps
    const TestRange gap_range = kTestInputVector[kGapCheckEntry].first;
    sparse_container::mapped_gap_generator gap1(map0, gap_range);
    ASSERT_TRUE(gap1->begin == gap_range.end);
    ASSERT_TRUE(gap1->end == kMaxRangeIndex);

    auto pre_gap = map0.find(gap_range);
    --pre_gap;
    TestRange prev_range = pre_gap->first;
    TestRange gap_check_range{gap_range};
    gap_check_range.begin -= 1;
    gap_check_range.end += 1;

    // Test the other way to get to a lower_bound gap (the first being at construction)
    sparse_container::mapped_gap_generator gap2(map0);
    gap2.seek_forward(gap_check_range);
    ASSERT_TRUE(gap2->begin == prev_range.end);
    ASSERT_TRUE(gap2->end == gap_range.begin);

    sparse_container::mapped_range_generator gen3(map0);
    sparse_container::mapped_gap_generator gap3(map0);
    sparse_container::intersection_generator intersect0(gen3, gap3);
    ASSERT_TRUE(intersect0->empty());
    sparse_container::intersection_generator intersect1(gap3, gen3);
    ASSERT_TRUE(intersect1->empty());

    // Create an "every other" map
    TestMap even_entries;
    TestMap odd_entries;
    bool even = true;
    for (const auto& value : sorted) {
        if (even)
            even_entries.insert(value);
        else
            odd_entries.insert(value);
        even = !even;
    }

    // Use the interestion of the even_entries ranges and map0 to fill in merged to have all entries
    TestMap merged(odd_entries);
    sparse_container::mapped_range_generator gen_even(even_entries);
    sparse_container::intersection_generator intersect2(gen_even, gen3);

    for (; !intersect2->empty(); ++intersect2) {
        auto pos = map0.find(*intersect2);
        ASSERT_TRUE(pos != map0.end());
        auto gap_pos = merged.find(*intersect2);
        ASSERT_TRUE(gap_pos == merged.end());

        merged.insert(*pos);
    }
    ASSERT_TRUE(MapRefMatches(merged, sorted));

    // Validate the "range -> range_gen" adapter
    TestRange ref_single{10, 21};
    TestRange ref_seek_valid{0, 0};  // Single range only advances is seek > range
    TestRange ref_seek_end{21, 22};
    sparse_container::single_range_generator single0(ref_single);
    ASSERT_TRUE(*single0 == ref_single);
    ++single0;
    ASSERT_TRUE(single0->empty());

    sparse_container::single_range_generator single1(ref_single);
    single1.seek_forward(ref_seek_valid);
    ASSERT_TRUE(*single1 == ref_single);
    single1.seek_forward(ref_seek_end);
    ASSERT_TRUE(single1->empty());
}
