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

class AccessState {
  public:
    using OpID = uint64_t;
    struct OpResult {
        // Used to identify when a state has been accessed  during the current operation
        // These fields are mutable to allow update during const traversal
        OpID op = OpID(0);

        // Needs to be a shared pointer as it is the value that will be copied to all nodes with a matching op tag
        std::shared_ptr<const AccessState> state;
    };

    // TODO: WIP: Remove default argument to *not* allow isolated inserted
    AccessState(const ResourceAccessState &state, const AccessState *insert_before = nullptr);
    // Treat prev and next as mutable...
    AccessState(const AccessState &other) : AccessState(other.state_, const_cast<AccessState *>(&other)) {}
    ~AccessState();

    const OpResult &GetResult() const { return result_; }
    const ResourceAccessState &GetState() const { return state_; }
    ResourceAccessState &GetState() { return state_; }

    // NOTE: Updates mutable member, though const
    void UpdateResult(OpResult &&from) const { result_ = std::move(from); }

    template <typename Detector>
    HazardResult DetectOp(OpID op_id, Detector &detector);

    template <typename Updater>
    void UpdateOp(OpID op_id, Updater &updater);

  private:
    // Treat the access state as a "static single assignment" of a ResourceAccessState, allowing safe sharing by many address
    // ranges, with the results of a given op
    // TODO: Make this const when we have constructors that can apply update operations during construction
    ResourceAccessState state_;

    // Linked list to allow global operations.
    // NOTE: *NOT* in range map order
    mutable const AccessState *prev_ = nullptr;  // Needed to support new global operations
    mutable const AccessState *next_ = nullptr;

    mutable OpResult result_;
};

using AccessStatePtr = std::shared_ptr<AccessState>;
using AccessStateConstPtr = std::shared_ptr<const AccessState>;

using ResourceAccessStateFunction = std::function<void(ResourceAccessState *)>;
using ResourceAccessRangeMap = sparse_container::range_map<ResourceAddress, AccessStatePtr>;
using ResourceRangeMergeIterator = sparse_container::parallel_iterator<ResourceAccessRangeMap, const ResourceAccessRangeMap>;

using RangeGenWrapper = sparse_container::range_gen_wrapper<ResourceAccessRange>;
using MapGapGen = sparse_container::mapped_gap_generator<ResourceAccessRangeMap>;
using MapRangeGen = sparse_container::mapped_range_generator<ResourceAccessRangeMap>;
using IntersectionAdvanceSecond = sparse_container::intersection_advance_policy_second;
using SingleRangeGen = sparse_container::single_range_generator<ResourceAccessRange>;

struct NoopBarrierAction {
    explicit NoopBarrierAction() {}
    void operator()(ResourceAccessState *access) const {}
    const bool layout_transition = false;
};

// This functor applies a collection of barriers, updating the "pending state" in each touched memory range, and optionally
// resolves the pending state. Suitable for processing Global memory barriers, or Subpass Barriers when the "final" barrier
// of a collection is known/present.
template <typename BarrierOp, typename OpVector = std::vector<BarrierOp>>
class ApplyBarrierOpsFunctor {
  public:
    using Iterator = ResourceAccessRangeMap::iterator;
    // Only called with a gap, and pos at the lower_bound(range)
    inline Iterator Infill(ResourceAccessRangeMap *accesses, const Iterator &pos, const ResourceAccessRange &range) const {
        if (!infill_default_) {
            return pos;
        }
        ResourceAccessState default_state;
        auto inserted = accesses->insert(pos, std::make_pair(range, std::make_shared<AccessState>(default_state)));
        return inserted;
    }

    void operator()(const Iterator &pos) const {
        auto &access_state = pos->second->GetState();
        for (const auto &op : barrier_ops_) {
            op(&access_state);
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
    void operator()(ResourceAccessState *access) const {
        assert(access);
        access->ApplyBarriers(barriers, true);
    }
    const std::vector<SyncBarrier> &barriers;
};

struct QueueTagOffsetBarrierAction {
    QueueTagOffsetBarrierAction(QueueId qid, ResourceUsageTag offset) : queue_id(qid), tag_offset(offset) {}
    void operator()(ResourceAccessState *access) const {
        access->OffsetTag(tag_offset);
        access->SetQueueId(queue_id);
    };
    QueueId queue_id;
    ResourceUsageTag tag_offset;
};

struct ApplyTrackbackStackAction {
    explicit ApplyTrackbackStackAction(const std::vector<SyncBarrier> &barriers_,
                                       const ResourceAccessStateFunction *previous_barrier_ = nullptr)
        : barriers(barriers_), previous_barrier(previous_barrier_) {}
    void operator()(ResourceAccessState *access) const {
        assert(access);
        assert(!access->HasPendingState());
        access->ApplyBarriers(barriers, false);
        // NOTE: We can use invalid tag, as these barriers do no include layout transitions (would assert in SetWrite)
        access->ApplyPendingBarriers(kInvalidTag);
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
    }

    void ResolvePreviousAccesses();
    void ResolveFromContext(const AccessContext &from);

    template <typename ResolveOp>
    void ResolveFromContext(ResolveOp &&resolve_op, const AccessContext &from_context,
                            const ResourceAccessState *infill_state = nullptr, bool recur_to_infill = false);
    template <typename ResolveOp, typename RangeGenerator>
    void ResolveFromContext(ResolveOp &&resolve_op, RangeGenerator range_gen, const AccessContext &from_context,
                            const ResourceAccessState *infill_state = nullptr, bool recur_to_infill = false);

    template <typename RangeGen>
    static ResourceAccessRangeMap InfillRangeGen(const RangeGen &gen_arg, const ResourceAccessStateFunction *previous_barrier,
                                                 const ResourceAccessState *infill_state);

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
    void ApplyToContext(const Action &barrier_action);

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

    template <typename Action>
    void ForAll(Action &&action);
    template <typename Action>
    void ConstForAll(Action &&action) const;
    template <typename Predicate>
    void EraseIf(Predicate &&pred);

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

    template <typename Action, typename RangeGen>
    void UpdateMemoryAccessState(const Action &action, RangeGen &range_gen);

  private:
    template <typename Action>
    static void UpdateMemoryAccessRangeState(ResourceAccessRangeMap &accesses, Action &action, const ResourceAccessRange &range);

    struct UpdateMemoryAccessStateFunctor {
        using Iterator = ResourceAccessRangeMap::iterator;
        Iterator Infill(ResourceAccessRangeMap *accesses, const Iterator &pos, const ResourceAccessRange &range) const;
        void operator()(const Iterator &pos) const;
        UpdateMemoryAccessStateFunctor(const AccessContext &context_, SyncStageAccessIndex usage_, SyncOrdering ordering_rule_,
                                       ResourceUsageTag tag_)
            : context(context_), usage_info(SyncStageAccess::UsageInfo(usage_)), ordering_rule(ordering_rule_), tag(tag_) {}
        const AccessContext &context;
        const SyncStageAccessInfoType &usage_info;
        const SyncOrdering ordering_rule;
        const ResourceUsageTag tag;
    };

    // Follow the context previous to access the access state, supporting "lazy" import into the context. Not intended for
    // subpass layout transition, as the pending state handling is more complex
    // TODO: See if returning the lower_bound would be useful from a performance POV -- look at the lower_bound overhead
    // Would need to add a "hint" overload to parallel_iterator::invalidate_[AB] call, if so.
    // NOTE: The use of the wrapper prevents recursive template expansion.
    ResourceAccessRangeMap ResolvePreviousAccessRangeGen(const RangeGenWrapper &gen_arg, const ResourceAccessState *infill_state,
                                                         const ResourceAccessStateFunction *previous_barrier = nullptr) const;

    template <typename RangeGen, typename BarrierAction>
    ResourceAccessRangeMap ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                 const ResourceAccessState *infill_state, bool recur_to_infill) const;
    template <typename Detector>
    HazardResult DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const;
    template <typename Detector, typename RangeGen>
    HazardResult DetectHazardGeneratedRanges(Detector &detector, RangeGen &range_gen, DetectOptions options) const;
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

    static void ResolveAccessMaps(const ResourceAccessRangeMap &from, ResourceAccessRangeMap &to);

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
};

// The semantics of the InfillUpdateOps of infill_update_range are slightly different than for the UpdateMemoryAccessState Action
// operations, as this simplifies the generic traversal.  So we wrap them in a semantics Adapter to get the same effect.
template <typename Action>
struct ActionToOpsAdapter {
    using Map = ResourceAccessRangeMap;
    using Range = typename Map::key_type;
    using Iterator = typename Map::iterator;
    using IndexType = typename Map::index_type;

    void infill(Map &accesses, const Iterator &pos, const Range &infill_range) const {
        // Combine Infill and update operations to make the generic implementation simpler
        Iterator infill = action.Infill(&accesses, pos, infill_range);
        if (infill == accesses.end()) return;  // Allow action to 'pass' on filling in the blanks

        // Need to apply the action to the Infill.  'infill_update_range' expect ops.infill to be completely done with
        // the infill_range, where as Action::Infill assumes the caller will apply the action() logic to the infill_range
        for (; infill != pos; ++infill) {
            assert(infill != accesses.end());
            action(infill);
        }
    }
    void update(const Iterator &pos) const {
        if (pos->second.use_count() > 1) {
            pos->second = std::make_shared<AccessState>(*pos->second);
        }
        action(pos);
    }
    const Action &action;
};

struct InfillOnlyOps {
    using Map = ResourceAccessRangeMap;
    using Range = typename Map::key_type;
    using Iterator = typename Map::iterator;

    void infill(Map &accesses, const Iterator &pos, const Range &infill_range) const {
        // TODO WIP interface need to make insertion into the AccessState linked list easy
        auto inserted = accesses.insert(pos, std::make_pair(infill_range, std::make_shared<AccessState>(infill_state)));
    }
    void update(const Iterator &pos) const {}
    const ResourceAccessState &infill_state;
};

template <typename Action>
void AccessContext::ApplyToContext(const Action &barrier_action) {
    // Note: Barriers do *not* cross context boundaries, applying to accessess within.... (at least for renderpass subpasses)
    UpdateMemoryAccessRangeState(access_state_map_, barrier_action, kFullRange);
}

template <typename Action>
void AccessContext::UpdateMemoryAccessRangeState(ResourceAccessRangeMap &accesses, Action &action,
                                                 const ResourceAccessRange &range) {
    ActionToOpsAdapter<Action> ops{action};
    infill_update_range(accesses, range, ops);
}

template <typename Action, typename RangeGen>
void AccessContext::UpdateMemoryAccessState(const Action &action, RangeGen &range_gen) {
    ActionToOpsAdapter<Action> ops{action};
    infill_update_rangegen(access_state_map_, range_gen, ops);
}

template <typename Action>
void AccessContext::ApplyUpdateAction(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type, const Action &action) {
    const std::optional<ImageRangeGen> &ref_range_gen = view_gen.GetRangeGen(gen_type);
    if (ref_range_gen) {
        ImageRangeGen range_gen(*ref_range_gen);
        UpdateMemoryAccessState(action, range_gen);
    }
}

template <typename Detector>
HazardResult AccessContext::DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const {
    sparse_container::single_range_generator range_gen(range);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}

template <typename Predicate>
void AccessContext::EraseIf(Predicate &&pred) {
    // Note: Don't forward, we don't want r-values moved, since we're going to make multiple calls.
    vvl::EraseIf(access_state_map_, pred);
}

template <typename ResolveOp>
void AccessContext::ResolveFromContext(ResolveOp &&resolve_op, const AccessContext &from_context,
                                       const ResourceAccessState *infill_state, bool recur_to_infill) {
    ResolveFromContext(resolve_op, SingleRangeGen(kFullRange), from_context, infill_state, recur_to_infill);
}

template <typename ResolveOp, typename RangeGenerator>
void AccessContext::ResolveFromContext(ResolveOp &&resolve_op, RangeGenerator range_gen, const AccessContext &from_context,
                                       const ResourceAccessState *infill_state, bool recur_to_infill) {
    ResourceAccessRangeMap resolve_from = from_context.ResolveAccessRangeGen(range_gen, resolve_op, infill_state, recur_to_infill);
    ResolveAccessMaps(resolve_from, access_state_map_);
}

template <typename RangeGen, typename BarrierAction>
ResourceAccessRangeMap AccessContext::ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                            const ResourceAccessState *infill_state, bool recur_to_infill) const {
    ResourceAccessRangeMap resolve_map;
    sparse_container::intersection_generator range_gen(gen_arg, MapRangeGen(access_state_map_), IntersectionAdvanceSecond());
    for (; !range_gen->empty(); ++range_gen) {
        const auto &pos = range_gen.get_second_gen().get_pos();
        ResourceAccessState access(pos->second->GetState());  // intentional copy
        barrier_action(&access);
        resolve_map.insert(resolve_map.end(), std::make_pair(pos->first, std::make_shared<AccessState>(access)));
    }

    // Infill if range goes passed both the current and resolve map prior contents
    if (recur_to_infill) {
        // This generator is only contains the ranges in gen_arg that are not present in the current map
        RangeGenWrapper recur_gen(sparse_container::intersection_generator(MapGapGen(access_state_map_), gen_arg));
        ResourceAccessStateFunction recur_barrier(std::ref(barrier_action));
        const ResourceAccessRangeMap prev_map = ResolvePreviousAccessRangeGen(recur_gen, infill_state, &recur_barrier);

        if (!prev_map.empty()) {
            // Resolves to second argument
            ResolveAccessMaps(prev_map, resolve_map);
        }
    }
    return resolve_map;
}
