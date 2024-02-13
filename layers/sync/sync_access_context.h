/*
 * Copyright (c) 2019-2023 Valve Corporation
 * Copyright (c) 2019-2023 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "state_tracker/render_pass_state.h"
#include "sync/sync_common.h"
#include "sync/sync_access_state.h"

namespace vvl {
class Buffer;
}  // namespace vvl

namespace syncval_state {
class ImageState;
class ImageViewState;
}  // namespace syncval_state
bool SimpleBinding(const BINDABLE &bindable);
VkDeviceSize ResourceBaseAddress(const vvl::Buffer &buffer);

// AccessState:
//
// Encapsulate the ResourceAccess state for a given range into a shareable object, with meta-data allowing:
//    * Single state assignment for all ranges with the same state
//    * Single execution for all operations (Detection or Update) for a given access state
//    * Global traversal of unique states, without traversing range map
//
// Theory of Operation:
//    For each non-empty range in an access context, a shared_ptr to the state for that range is stored.  The goal
//    is that for each unique state, only a single copy exists. (Note: The trade off for when, where, and how often
//    duplicate states are combined is TBD.  Likely candidates include at the end of each ExecutionContext or AccessContext
//    processing (for example after EndCommandBuffer, EndRenderPass, or after each queue batch completes).
//
// Traversal Operations
//    When an access context Detect or Update operation traverses a tree, that operation is given a unique OpID (monotonically
//    increasing integer).  During traversal, if a range state does not match the current OpID, the action (Detect or Update) is
//    performed, and the AccessState update with the results (if any).  For Detect operations with *no* hazard detected, the
//    AccessState is updated to refelct the current OpID.  Given that hazards terminate traversal, no updated state is needed.
//    For Update operations,
//        * the AccessState is duplicated.
//        * the Update operation is performed on the duplicate
//        * the original AccessState result is updated to refer to the updated state
//        * the range's mapped value is updated to with the shared pointer to the updated state.
//
//    During traversal, if a range is encountered with an AccessState matching the OpID, the operations is
//    treated as already performed.  For Detection traversal, the result is assumed to be "no hazard" since detection of a hazard
//    terminates traversal.  For Update traversal the state associated with the result is taken as the new mapped_type value, and
//    traversal continues.
//
// Global Operation Support
//    Several (especially pre sync2) operations affect all entries in an AccessContext.  To optimize this, all AccessState objects
//    are organized in a linked list.  The AccessContext range map's begin will point to an arbitrary location within this list,
//    thus global operations are performed on begin() as well as walking the list in previous and next directions.
//
// Composite/Pending operations
//    Note that for Vulkan operations that imply more than one Update, each specific Update will be treated independently.  This
//    implies *specifically* that intermediatate states (such as those with pending_ field updates, resolutions) will be SSA
//    copies.
//
// NOTE:
//    Need to check performance on singly referenced AccessState, to see if modification in place is needed, and
//    assessment the complexity, risk, and performance gains of allowing modification in place for those operations.
//

class AccessState;
class AccessContext;
using AccessStatePtr = std::shared_ptr<AccessState>;
using AccessStateConstPtr = std::shared_ptr<const AccessState>;

class AccessState {
  public:
    using OpID = uint64_t;
    struct OpResult {
        // Used to identify when a state has been accessed  during the current operation
        // These fields are mutable to allow update during const traversal
        OpID op = OpID(0);

        // Needs to be a shared pointer as it is the value that will be copied to all nodes with a matching op tag
        AccessStateConstPtr state = nullptr;
        std::optional<std::map<const AccessState *, AccessStateConstPtr>> resolve_cache;

        void InitResolve(OpID op_id) {
            op = op_id;
            state.reset();
            resolve_cache.emplace();
        }
        AccessStateConstPtr LookupResolve(const AccessStateConstPtr &source_ptr) {
            assert(resolve_cache);
            auto &cache = resolve_cache.value();
            auto find_it = cache.find(source_ptr.get());
            if (find_it != cache.end()) {
                return find_it->second;
            }
            return AccessStateConstPtr();
        }

        void CacheResolve(const AccessStateConstPtr &source, const AccessStateConstPtr &resolved) {
            assert(resolve_cache);
            resolve_cache->insert(std::make_pair(source.get(), resolved));
        }
    };

    class OpContext {
      public:
        const AccessState *InsertPoint() const;
        const AccessStateConstPtr &DefaultState() const;
        OpID Op() const;
        OpContext(const AccessContext &access_context);
        OpContext(const OpContext &other_context);

      protected:
        static std::atomic<OpID> next_op_;  // Ops must be unique over the life of a device (at least)
        const AccessContext &context_;
        const OpID op_id_;
    };

    AccessState() : state_() {}

    AccessState(const ResourceAccessState &from_state, const AccessState *insert_point) : state_(from_state) {
        InsertMutable(insert_point);
    }
    AccessState(const AccessState &from, const AccessState *insert_point) : AccessState(from.state_, insert_point) {}

    template <typename Operation>
    AccessState(const ResourceAccessState &from_state, const AccessState *insert_point, const Operation &op)
        : state_(from_state, op) {
        InsertMutable(insert_point);
    }
    template <typename Operation>
    AccessState(const AccessState &from, const AccessState *insert_point, const Operation &op)
        : AccessState(from.state_, insert_point, op) {}

    template <typename Updater>
    AccessStateConstPtr SSAUpdate(const OpContext &op, const Updater &updater) const;

    AccessStateConstPtr SSAResolve(const AccessStateConstPtr &other_access_state, const OpContext &op) const;

    // Treat prev and next as mutable...
    // AccessState(const AccessState &other) : AccessState(other.state_, &other) {}
    ~AccessState();

    const OpResult &GetResult() const { return result_; }
    const ResourceAccessState &GetState() const { return state_; }

    // NOTE: Updates mutable member, though const
    void UpdateResult(OpResult &&from) const { result_ = std::move(from); }

    const AccessState *GetPrev() const { return prev_; }

    template <typename Transform>
    struct TransformUpdater {
        AccessStateConstPtr operator()(const AccessState::OpContext &op_context, const AccessState &input) const {
            return std::make_shared<const AccessState>(input, op_context.InsertPoint(), transform);
        }
        TransformUpdater(const Transform &func_) : transform(func_) {}
        const Transform &transform;
    };

  private:
    void InsertMutable(const AccessState *before);
    // Treat the access state as a "static single assignment" of a ResourceAccessState, allowing safe sharing by many address
    // ranges, with the results of a given op
    // TODO: Make this const when we have constructors that can apply update operations during construction
    const ResourceAccessState state_;

    // Temporary result for current operation on AccessContext
    mutable OpResult result_;

    // Linked list to allow global operations.
    // NOTE: *NOT* in range map order
    mutable const AccessState *prev_ = nullptr;  // Needed to support new global operations
    mutable const AccessState *next_ = nullptr;
};

using ResourceAccessRangeMap = sparse_container::range_map<ResourceAddress, AccessStateConstPtr>;
using ResourceRangeMergeIterator = sparse_container::parallel_iterator<ResourceAccessRangeMap, const ResourceAccessRangeMap>;

using RangeGenWrapper = sparse_container::range_gen_wrapper<ResourceAccessRange>;
using MapGapGen = sparse_container::mapped_gap_generator<ResourceAccessRangeMap>;
using MapRangeGen = sparse_container::mapped_range_generator<ResourceAccessRangeMap>;
using IntersectionAdvanceSecond = sparse_container::intersection_advance_policy_second;
using SingleRangeGen = sparse_container::single_range_generator<ResourceAccessRange>;

struct ApplyTrackbackStackAction {
    explicit ApplyTrackbackStackAction(const std::vector<SyncBarrier> &barriers_,
                                       const ResourceAccessStateFunction *previous_barrier_ = nullptr)
        : barriers(barriers_), previous_barrier(previous_barrier_) {}
    void operator()(ResourceAccessState &access) const {
        assert(!access.HasPendingState());
        access.ApplyBarriers(barriers, false);
        // NOTE: We can use invalid tag, as these barriers do no include layout transitions (would assert in SetWrite)
        access.ApplyPendingBarriers(kInvalidTag);
        if (previous_barrier) {
            assert(bool(*previous_barrier));
            (*previous_barrier)(access);
        }
    }
    const std::vector<SyncBarrier> &barriers;
    const ResourceAccessStateFunction *previous_barrier;
};

template <typename SubpassNode>
struct SubpassBarrierTrackback {
    std::vector<SyncBarrier> barriers;
    const SubpassNode *source_subpass = nullptr;
    SubpassBarrierTrackback() = default;
    SubpassBarrierTrackback(const SubpassBarrierTrackback &) = default;
    SubpassBarrierTrackback(const SubpassNode *source_subpass_, VkQueueFlags queue_flags_,
                            const std::vector<const VkSubpassDependency2 *> &subpass_dependencies_)
        : barriers(), source_subpass(source_subpass_) {
        barriers.reserve(subpass_dependencies_.size());
        for (const VkSubpassDependency2 *dependency : subpass_dependencies_) {
            assert(dependency);
            barriers.emplace_back(queue_flags_, *dependency);
        }
    }
    SubpassBarrierTrackback(const SubpassNode *source_subpass_, const SyncBarrier &barrier_)
        : barriers(1, barrier_), source_subpass(source_subpass_) {}
    SubpassBarrierTrackback &operator=(const SubpassBarrierTrackback &) = default;
};

class AttachmentViewGen {
  public:
    enum Gen { kViewSubresource = 0, kRenderArea = 1, kDepthOnlyRenderArea = 2, kStencilOnlyRenderArea = 3, kGenSize = 4 };
    AttachmentViewGen(const syncval_state::ImageViewState *image_view, const VkOffset3D &offset, const VkExtent3D &extent);
    AttachmentViewGen(const AttachmentViewGen &other) = default;
    AttachmentViewGen(AttachmentViewGen &&other) = default;
    const syncval_state::ImageViewState *GetViewState() const { return view_; }
    const std::optional<ImageRangeGen> &GetRangeGen(Gen type) const;
    bool IsValid() const { return gen_store_[Gen::kViewSubresource].has_value(); }
    Gen GetDepthStencilRenderAreaGenType(bool depth_op, bool stencil_op) const;

  private:
    const syncval_state::ImageViewState *view_ = nullptr;
    VkImageAspectFlags view_mask_ = 0U;
    std::array<std::optional<ImageRangeGen>, Gen::kGenSize> gen_store_;
};

using AttachmentViewGenVector = std::vector<AttachmentViewGen>;

struct SyncImageMemoryBarrier {
    using ImageState = syncval_state::ImageState;
    using Image = std::shared_ptr<const ImageState>;

    Image image;
    uint32_t index;
    SyncBarrier barrier;
    VkImageLayout old_layout;
    VkImageLayout new_layout;
    VkImageSubresourceRange range;

    bool IsLayoutTransition() const { return old_layout != new_layout; }
    const VkImageSubresourceRange &Range() const { return range; };
    const ImageState *GetState() const { return image.get(); }
    SyncImageMemoryBarrier(const Image &image_, uint32_t index_, const SyncBarrier &barrier_, VkImageLayout old_layout_,
                           VkImageLayout new_layout_, const VkImageSubresourceRange &subresource_range_)
        : image(image_),
          index(index_),
          barrier(barrier_),
          old_layout(old_layout_),
          new_layout(new_layout_),
          range(subresource_range_) {}
    SyncImageMemoryBarrier() = default;
};

class AccessContext {
  public:
    using ImageState = syncval_state::ImageState;
    using ImageViewState = syncval_state::ImageViewState;
    enum DetectOptions : uint32_t {
        kDetectPrevious = 1U << 0,
        kDetectAsync = 1U << 1,
        kDetectAll = (kDetectPrevious | kDetectAsync)
    };

    enum UpdateOptions : uint32_t {
        kExtantOnly = 0,
        kResolvePrevious = 1U << 0,
        kInfillDefault = 1U << 1,
        kUpdateAll = (kResolvePrevious | kInfillDefault)
    };

    static bool UpdateOptionResolvePrevious(UpdateOptions options) {
        return static_cast<uint32_t>(options) & UpdateOptions::kResolvePrevious;
    }
    static bool UpdateOptionInfillDefault(UpdateOptions options) {
        return static_cast<uint32_t>(options) & UpdateOptions::kInfillDefault;
    }

    using TrackBack = SubpassBarrierTrackback<AccessContext>;

    HazardResult DetectHazard(const vvl::Buffer &buffer, SyncStageAccessIndex usage_index, const ResourceAccessRange &range) const;
    HazardResult DetectHazard(const ImageState &image, SyncStageAccessIndex current_usage,
                              const VkImageSubresourceRange &subresource_range, bool is_depth_sliced) const;
    HazardResult DetectHazard(const ImageViewState &image_view, SyncStageAccessIndex current_usage) const;
    HazardResult DetectHazard(const ImageRangeGen &ref_range_gen, SyncStageAccessIndex current_usage,
                              SyncOrdering ordering_rule = SyncOrdering::kOrderingNone) const;
    HazardResult DetectHazard(const ImageViewState &image_view, const VkOffset3D &offset, const VkExtent3D &extent,
                              SyncStageAccessIndex current_usage, SyncOrdering ordering_rule) const;
    HazardResult DetectHazard(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                              SyncStageAccessIndex current_usage, SyncOrdering ordering_rule) const;
    HazardResult DetectHazard(const ImageState &image, const VkImageSubresourceRange &subresource_range, const VkOffset3D &offset,
                              const VkExtent3D &extent, bool is_depth_sliced, SyncStageAccessIndex current_usage,
                              SyncOrdering ordering_rule = SyncOrdering::kOrderingNone) const;

    HazardResult DetectImageBarrierHazard(const ImageState &image, const VkImageSubresourceRange &subresource_range,
                                          VkPipelineStageFlags2KHR src_exec_scope, const SyncStageAccessFlags &src_access_scope,
                                          QueueId queue_id, const AccessContext &first_scope, ResourceUsageTag scope_tag,
                                          DetectOptions options) const;
    HazardResult DetectImageBarrierHazard(const AttachmentViewGen &attachment_view, const SyncBarrier &barrier,
                                          DetectOptions options) const;
    HazardResult DetectImageBarrierHazard(const ImageState &image, VkPipelineStageFlags2KHR src_exec_scope,
                                          const SyncStageAccessFlags &src_access_scope,
                                          const VkImageSubresourceRange &subresource_range, DetectOptions options) const;
    HazardResult DetectImageBarrierHazard(const SyncImageMemoryBarrier &image_barrier) const;
    HazardResult DetectSubpassTransitionHazard(const TrackBack &track_back, const AttachmentViewGen &attach_view) const;

    HazardResult DetectFirstUseHazard(QueueId queue_id, const ResourceUsageRange &tag_range,
                                      const AccessContext &access_context) const;

    const TrackBack &GetDstExternalTrackBack() const { return dst_external_; }
    void Reset() {
        prev_.clear();
        prev_by_subpass_.clear();
        async_.clear();
        src_external_ = nullptr;
        dst_external_ = TrackBack();
        start_tag_ = ResourceUsageTag();
        access_state_map_.clear();
        default_state_ = std::make_shared<const AccessState>();
    }

    void ResolvePreviousAccesses();
    void ResolveFromContext(const AccessContext &from);

    template <typename ResolveOp>
    void ResolveFromContext(ResolveOp &&resolve_op, const AccessContext &from_context, bool infill_default = false,
                            bool recur_to_infill = false);
    template <typename ResolveOp, typename RangeGenerator>
    void ResolveFromContext(const ResolveOp &resolve_op, const RangeGenerator &range_gen, const AccessContext &from_context,
                            bool infill_default = false, bool recur_to_infill = false);

    template <typename RangeGen>
    static ResourceAccessRangeMap InfillRangeGen(const RangeGen &gen_arg, const ResourceAccessStateFunction *previous_barrier);

    void UpdateAccessState(const vvl::Buffer &buffer, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           const ResourceAccessRange &range, ResourceUsageTag tag);
    void UpdateAccessState(const ImageState &image, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           const VkImageSubresourceRange &subresource_range, const ResourceUsageTag &tag);
    void UpdateAccessState(const ImageState &image, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           const VkImageSubresourceRange &subresource_range, const VkOffset3D &offset, const VkExtent3D &extent,
                           ResourceUsageTag tag);
    void UpdateAccessState(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type, SyncStageAccessIndex current_usage,
                           SyncOrdering ordering_rule, ResourceUsageTag tag);
    void UpdateAccessState(const ImageViewState &image_view, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           const VkOffset3D &offset, const VkExtent3D &extent, ResourceUsageTag tag);
    void UpdateAccessState(const ImageViewState &image_view, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           ResourceUsageTag tag);
    void UpdateAccessState(const ImageRangeGen &range_gen, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           ResourceUsageTag tag);
    void UpdateAccessState(ImageRangeGen &range_gen, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                           ResourceUsageTag tag);
    void ResolveChildContexts(const std::vector<AccessContext> &contexts);

    void ImportAsyncContexts(const AccessContext &from);
    void ClearAsyncContexts() { async_.clear(); }
    template <typename Action>
    void ApplyUpdateAction(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type, const Action &action);
    template <typename Action>
    void ApplyToContext(const Action &barrier_action, UpdateOptions options);

    AccessContext(uint32_t subpass, VkQueueFlags queue_flags, const std::vector<SubpassDependencyGraphNode> &dependencies,
                  const std::vector<AccessContext> &contexts, const AccessContext *external_context);

    AccessContext() { Reset(); }
    AccessContext(const AccessContext &copy_from) = default;
    void Trim();
    void TrimAndClearFirstAccess();
    void AddReferencedTags(ResourceUsageTagSet &referenced) const;

    ResourceAccessRangeMap &GetAccessStateMap() { return access_state_map_; }
    const ResourceAccessRangeMap &GetAccessStateMap() const { return access_state_map_; }
    const TrackBack *GetTrackBackFromSubpass(uint32_t subpass) const {
        if (subpass == VK_SUBPASS_EXTERNAL) {
            return src_external_;
        } else {
            assert(subpass < prev_by_subpass_.size());
            return prev_by_subpass_[subpass];
        }
    }

    void SetStartTag(ResourceUsageTag tag) { start_tag_ = tag; }
    ResourceUsageTag StartTag() const { return start_tag_; }

#if 0
    template <typename Predicate>
    void EraseIf(Predicate &&pred);
#endif

    // For use during queue submit building up the QueueBatchContext AccessContext for validation, otherwise clear.
    void AddAsyncContext(const AccessContext *context, ResourceUsageTag tag);

    class AsyncReference {
      public:
        AsyncReference(const AccessContext &async_context, ResourceUsageTag async_tag)
            : context_(&async_context), tag_(async_tag) {}
        const AccessContext &Context() const { return *context_; }
        // For RenderPass time validation this is "start tag", for QueueSubmit, this is the earliest
        // unsynchronized tag for the Queue being tested against (max synchrononous + 1, perhaps)
        ResourceUsageTag StartTag() const;

      protected:
        const AccessContext *context_;
        ResourceUsageTag tag_;  // Start of open ended asynchronous range
    };

    void ApplyAcquireWait(ResourceUsageTag present_tag, ResourceUsageTag acquire_tag);
    void ApplyTaggedWait(QueueId queue_id, ResourceUsageTag tag);

    template <typename Action, typename RangeGen>
    void UpdateMemoryAccessState(const Action &action, const RangeGen &range_gen, UpdateOptions options);
    const AccessStateConstPtr &DefaultAccessState() const { return default_state_; }

  private:
    template <typename Updater>
    void ForAll(Updater &&updater);

    template <typename Action>
    void ConstForAll(Action &&action) const;

    AccessState::OpContext MakeOpContext() const { return AccessState::OpContext(*this); }
    template <typename Action>
    void UpdateMemoryAccessRangeState(Action &action, const ResourceAccessRange &range, UpdateOptions options);
    template <typename Predicate>
    void ApplyPredicatedWait(const Predicate &predicate);

    struct UpdateMemoryAccessStateFunctor {
        void operator()(ResourceAccessState &) const;
        UpdateMemoryAccessStateFunctor(SyncStageAccessIndex usage_, SyncOrdering ordering_rule_, ResourceUsageTag tag_)
            : usage_info(SyncStageAccess::UsageInfo(usage_)), ordering_rule(ordering_rule_), tag(tag_) {}
        const SyncStageAccessInfoType &usage_info;
        const SyncOrdering ordering_rule;
        const ResourceUsageTag tag;
    };

    // Follow the context previous to access the access state, supporting "lazy" import into the context. Not intended for
    // subpass layout transition, as the pending state handling is more complex
    // TODO: See if returning the lower_bound would be useful from a performance POV -- look at the lower_bound overhead
    // Would need to add a "hint" overload to parallel_iterator::invalidate_[AB] call, if so.
    // NOTE: The use of the wrapper prevents recursive template expansion.
    ResourceAccessRangeMap ResolvePreviousAccessRangeGen(const RangeGenWrapper &gen_arg, const AccessState::OpContext &op,
                                                         const ResourceAccessStateFunction *previous_barrier = nullptr) const;

    template <typename RangeGen, typename BarrierAction>
    ResourceAccessRangeMap ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                 const AccessState::OpContext &op, bool recur_to_infill) const;
    template <typename Detector>
    HazardResult DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const;
    template <typename Detector, typename RangeGen>
    HazardResult DetectHazardGeneratedRanges(Detector &detector, RangeGen &range_gen, DetectOptions options) const;
    template <typename RangeGen>
    void InfillGapRangeGen(const RangeGen &gen_arg, const ResourceAccessStateFunction *infill_op);
    template <typename Detector, typename RangeGen>
    HazardResult DetectHazardGeneratedRanges(Detector &detector, const RangeGen &range_gen, DetectOptions options) const {
        RangeGen mutable_gen(range_gen);
        return DetectHazardGeneratedRanges<Detector, RangeGen>(detector, mutable_gen, options);
    }
    template <typename Detector>
    HazardResult DetectHazard(Detector &detector, const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                              DetectOptions options) const;
    template <typename Detector>
    HazardResult DetectHazard(Detector &detector, const ImageState &image, const VkImageSubresourceRange &subresource_range,
                              const VkOffset3D &offset, const VkExtent3D &extent, bool is_depth_sliced,
                              DetectOptions options) const;
    template <typename Detector>
    HazardResult DetectHazard(Detector &detector, const ImageState &image, const VkImageSubresourceRange &subresource_range,
                              bool is_depth_sliced, DetectOptions options) const;

    static void ResolveAccessMaps(const AccessState::OpContext &op, const ResourceAccessRangeMap &from, ResourceAccessRangeMap &to);

    template <typename Detector, typename RangeGen>
    HazardResult DetectAsyncHazard(const Detector &detector, const RangeGen &const_range_gen, ResourceUsageTag async_tag) const;

    template <typename NormalizeOp>
    void Trim(NormalizeOp &&normalize);

    ResourceAccessRangeMap access_state_map_;
    std::vector<TrackBack> prev_;
    std::vector<TrackBack *> prev_by_subpass_;
    // These contexts *must* have the same lifespan as this context, or be cleared, before the referenced contexts can expire
    std::vector<AsyncReference> async_;
    TrackBack *src_external_;
    TrackBack dst_external_;
    ResourceUsageTag start_tag_;
    AccessStateConstPtr default_state_;
    // Get is preincrement s.t. 0 is an invalid OpID.
};
struct AccessStateStats {
    static constexpr size_t kMismatchCount = static_cast<size_t>(MismatchReason::COUNT);
    static constexpr size_t kNoneLimit = static_cast<size_t>(MismatchReason::NONE_LIMIT);
    static size_t total_entries;
    static size_t reused_entries;
    static std::array<size_t, kMismatchCount> total_compare_stats;
    std::array<size_t, kMismatchCount> total_compare_copy;
    std::array<size_t, kMismatchCount> compare_stats;
    std::set<AccessStateConstPtr> access_states;
    std::set<ResourceAccessState> resource_states;
    void Gather(const ResourceAccessRangeMap &map) {
        access_states.clear();
        resource_states.clear();

        for (auto &s : compare_stats) s = 0;

        for (const auto &elem : map) {
            total_entries++;
            auto access_insert = access_states.insert(elem.second);
            if (!access_insert.second) {
                reused_entries++;
                continue;
            }
            size_t min_mismatch = kMismatchCount;
            const auto &state = elem.second->GetState();
            for (const auto &uniq : access_states) {
                if (uniq == elem.second) continue;
                size_t comparison = static_cast<size_t>(state.Compare(uniq->GetState()));
                if (comparison < min_mismatch) {
                    min_mismatch = comparison;
                    if (comparison < kNoneLimit) break;
                }
            }

            if (min_mismatch < kMismatchCount) {
                compare_stats[min_mismatch]++;
            } else {
                assert(access_states.size() == 1);
            }
        }
        std::array<float, kMismatchCount> total_compare_rates;
        for (size_t i = 0; i < kMismatchCount; i++) {
            total_compare_stats[i] += compare_stats[i];
            total_compare_copy[i] = total_compare_stats[i];
            total_compare_rates[i] = 100.F * static_cast<float>(total_compare_stats[i]) / static_cast<float>(total_entries);
        }
        float resuse_rate = 100.F * (float)reused_entries / (float)total_entries;
    }
};
struct NoopBarrierAction {
    explicit NoopBarrierAction() {}
    void operator()(ResourceAccessState &) const {}
    const bool layout_transition = false;
};

// This functor applies a collection of barriers, updating the "pending state" in each touched memory range, and optionally
// resolves the pending state. Suitable for processing Global memory barriers, or Subpass Barriers when the "final" barrier
// of a collection is known/present.
template <typename BarrierOp, typename OpVector = std::vector<BarrierOp>>
class ApplyBarrierOpsFunctor {
  public:
    void operator()(ResourceAccessState &access_state) const {
        for (const auto &op : barrier_ops_) {
            op(access_state);
        }
        if (resolve_) {
            // If this is the last (or only) batch, we can do the pending resolve as the last step in this operation to avoid
            // another walk
            access_state.ApplyPendingBarriers(tag_);
        }
    }

    // A valid tag is required IFF layout_transition is true, as transitions are write ops
    ApplyBarrierOpsFunctor(bool resolve, typename OpVector::size_type size_hint, ResourceUsageTag tag)
        : resolve_(resolve), infill_default_(false), barrier_ops_(), tag_(tag) {
        barrier_ops_.reserve(size_hint);
    }
    void EmplaceBack(const BarrierOp &op) {
        barrier_ops_.emplace_back(op);
        infill_default_ |= op.layout_transition;
    }
    AccessContext::UpdateOptions UpdateOptions() const {
        // TODO: Revisit these... while compatible with the pre-refactor behavior not recurring is probably creating false
        //       negatives, as infill without resolve prev will overwrite
        if (infill_default_) {
            return AccessContext::UpdateOptions::kInfillDefault;
        }
        return AccessContext::UpdateOptions::kExtantOnly;
    }

  private:
    bool resolve_;
    bool infill_default_;
    OpVector barrier_ops_;
    const ResourceUsageTag tag_;
};

// This functor applies a single barrier, updating the "pending state" in each touched memory range, but does not
// resolve the pendinging state. Suitable for processing Image and Buffer barriers from PipelineBarriers or Events
template <typename BarrierOp>
class ApplyBarrierFunctor : public ApplyBarrierOpsFunctor<BarrierOp, small_vector<BarrierOp, 1>> {
    using Base = ApplyBarrierOpsFunctor<BarrierOp, small_vector<BarrierOp, 1>>;

  public:
    ApplyBarrierFunctor(const BarrierOp &barrier_op) : Base(false, 1, kInvalidTag) { Base::EmplaceBack(barrier_op); }
};

// This functor resolves the pendinging state.
class ResolvePendingBarrierFunctor : public ApplyBarrierOpsFunctor<NoopBarrierAction, small_vector<NoopBarrierAction, 1>> {
    using Base = ApplyBarrierOpsFunctor<NoopBarrierAction, small_vector<NoopBarrierAction, 1>>;

  public:
    ResolvePendingBarrierFunctor(ResourceUsageTag tag) : Base(true, 0, tag) {}
};

struct ApplySubpassTransitionBarriersAction {
    explicit ApplySubpassTransitionBarriersAction(const std::vector<SyncBarrier> &barriers_) : barriers(barriers_) {}
    void operator()(ResourceAccessState &access) const { access.ApplyBarriers(barriers, true); }
    const std::vector<SyncBarrier> &barriers;
};

struct QueueTagOffsetBarrierAction {
    QueueTagOffsetBarrierAction(QueueId qid, ResourceUsageTag offset) : queue_id(qid), tag_offset(offset) {}
    void operator()(ResourceAccessState &access) const {
        access.OffsetTag(tag_offset);
        access.SetQueueId(queue_id);
    };
    QueueId queue_id;
    ResourceUsageTag tag_offset;
};

template <typename UpdateAction>
class SSAUpdateOperations {
  public:
    using Map = ResourceAccessRangeMap;
    using Range = typename Map::key_type;
    using Iterator = typename Map::iterator;

    void update(const Iterator &pos) const {
        auto &access_state = pos->second;
        access_state = access_state->SSAUpdate(op_context_, updater_);
    }

    void infill(Map &accesses, const Iterator &pos, const Range &infill_range) const {
        if (infill_state_) {
            accesses.insert(pos, std::make_pair(infill_range, infill_state_));
        }
    }

    SSAUpdateOperations(const UpdateAction &update_action, const AccessState::OpContext &op_context, bool infill_default)
        : updater_(update_action), op_context_(op_context), infill_state_() {
        if (infill_default) {
            const AccessStateConstPtr &default_state = op_context_.DefaultState();
            infill_state_ = std::make_shared<const AccessState>(*default_state, default_state.get(), updater_.transform);
        }
    }

  private:
    const AccessState::TransformUpdater<UpdateAction> updater_;
    const AccessState::OpContext &op_context_;
    AccessStateConstPtr infill_state_;
};

template <typename Action>
void AccessContext::ApplyToContext(const Action &barrier_action, const UpdateOptions options) {
    // Note: Barriers do *not* cross context boundaries, applying to accessess within.... (at least for renderpass subpasses)
    UpdateMemoryAccessRangeState(barrier_action, kFullRange, options);
}

template <typename Action>
void AccessContext::UpdateMemoryAccessRangeState(Action &action, const ResourceAccessRange &range, UpdateOptions options) {
    UpdateMemoryAccessState(action, sparse_container::single_range_generator(range), options);
}

template <typename Action, typename RangeGen>
void AccessContext::UpdateMemoryAccessState(const Action &action, const RangeGen &range_gen, UpdateOptions options) {
    // Resolve if required
    if (UpdateOptionResolvePrevious(options) && (prev_.size() > 0)) {
        // Only resolves in the intersection of range_gen and the gaps in access_state_map_
        auto prev_map = ResolvePreviousAccessRangeGen(range_gen, MakeOpContext(), nullptr);
        if (!prev_map.empty()) {
            sparse_container::merge_non_intersecting(access_state_map_, prev_map.begin(), prev_map.end(),
                                                     sparse_container::assign_policy_move());
        }
    }

    SSAUpdateOperations update_operations(action, MakeOpContext(), UpdateOptionInfillDefault(options));
    RangeGen gen_copy(range_gen);
    sparse_container::infill_update_rangegen(access_state_map_, gen_copy, update_operations);
}

template <typename Action>
void AccessContext::ApplyUpdateAction(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type, const Action &action) {
    const std::optional<ImageRangeGen> &ref_range_gen = view_gen.GetRangeGen(gen_type);
    if (ref_range_gen) {
        ImageRangeGen range_gen(*ref_range_gen);
        UpdateMemoryAccessState(action, range_gen, action.UpdateOptions());
    }
}

template <typename Detector>
HazardResult AccessContext::DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const {
    sparse_container::single_range_generator range_gen(range);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}

#if 0
template <typename Predicate>
void AccessContext::EraseIf(Predicate &&pred) {
    // Note: Don't forward, we don't want r-values moved, since we're going to make multiple calls.
    vvl::EraseIf(access_state_map_, pred);
}
#endif

template <typename ResolveOp>
void AccessContext::ResolveFromContext(ResolveOp &&resolve_op, const AccessContext &from_context, bool infill_default,
                                       bool recur_to_infill) {
    ResolveFromContext(resolve_op, SingleRangeGen(kFullRange), from_context, infill_default, recur_to_infill);
}

template <typename ResolveOp, typename RangeGenerator>
void AccessContext::ResolveFromContext(const ResolveOp &resolve_op, const RangeGenerator &range_gen,
                                       const AccessContext &from_context, bool infill_default, bool recur_to_infill) {
    ResourceAccessRangeMap resolve_from =
        from_context.ResolveAccessRangeGen(range_gen, resolve_op, MakeOpContext(), recur_to_infill);
    ResolveAccessMaps(MakeOpContext(), resolve_from, access_state_map_);
    if (infill_default) {
        auto resolve_wrapper = [&resolve_op](ResourceAccessState &access) { return resolve_op(access); };
        ResourceAccessStateFunction resolve_fun(resolve_wrapper);
        InfillGapRangeGen(range_gen, &resolve_fun);
    }
}

template <typename RangeGen, typename BarrierAction>
ResourceAccessRangeMap AccessContext::ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                            const AccessState::OpContext &op, bool recur_to_infill) const {
    ResourceAccessRangeMap resolve_map;
    sparse_container::intersection_generator range_gen(gen_arg, MapRangeGen(access_state_map_));
    AccessState::TransformUpdater barrier_updater(barrier_action);
    for (; !range_gen->empty(); ++range_gen) {
        const auto &pos = range_gen.get_second_gen().get_pos();
        AccessStateConstPtr resolved_state = pos->second->SSAUpdate(op, barrier_updater);
        resolve_map.insert(resolve_map.end(), std::make_pair(*range_gen, std::move(resolved_state)));
    }

    // Infill if range goes passed both the current and resolve map prior contents
    if (recur_to_infill && (prev_.size() > 0)) {
        // This generator is only contains the ranges in gen_arg that are not present in the current map
        RangeGenWrapper recur_gen(sparse_container::intersection_generator(MapGapGen(access_state_map_), gen_arg));
        ResourceAccessStateFunction recur_barrier(std::ref(barrier_action));
        // ResolvePrevious constructs it's on OpContext's so no need to do so before calling
        const ResourceAccessRangeMap prev_map = ResolvePreviousAccessRangeGen(recur_gen, op, &recur_barrier);

        if (!prev_map.empty()) {
            // Resolves to last argument, and needs it's own op context
            ResolveAccessMaps(AccessState::OpContext(op), prev_map, resolve_map);
        }
    }
    return resolve_map;
}

template <typename RangeGen>
void AccessContext::InfillGapRangeGen(const RangeGen &gen_arg, const ResourceAccessStateFunction *infill_op) {
    sparse_container::mapped_gap_generator gap_gen(access_state_map_);
    sparse_container::intersection_generator infill_gen(gen_arg, gap_gen);
    using ValueType = ResourceAccessRangeMap::value_type;
    AccessStateConstPtr infill_state;
    if (infill_op) {
        infill_state = std::make_shared<const AccessState>(*default_state_, default_state_.get(), *infill_op);
    } else {
        infill_state = default_state_;
    }

    for (; infill_gen->non_empty(); ++infill_gen) {
        // Inserting before the gap in access_state_map_, the won't confuse the generator when it advances, as
        // pos is after the current_range in the generator, and insertion doesn't invalidate iterators
        access_state_map_.insert(infill_gen.get_second_gen().get_pos(), std::make_pair(*infill_gen, infill_state));
    }
}

template <typename Updater>
AccessStateConstPtr AccessState::SSAUpdate(const OpContext &op, const Updater &updater) const {
    AccessStateConstPtr update;
    if (op.Op() == result_.op) {
        update = result_.state;
    } else {
        update = updater(op, *this);
        UpdateResult({op.Op(), update});
    }
    return update;
}
