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

class AccessState;
class AccessContext;
using AccessStatePtr = std::shared_ptr<AccessState>;
using AccessStateConstPtr = std::shared_ptr<const AccessState>;

namespace vvl {
class Buffer;
}  // namespace vvl

namespace syncval_state {
class ImageState;
class ImageViewState;
}  // namespace syncval_state

bool SimpleBinding(const BINDABLE &bindable);
VkDeviceSize ResourceBaseAddress(const vvl::Buffer &buffer);

using ResourceAccessRangeMap = sparse_container::range_map<ResourceAddress, AccessStateConstPtr>;
using ResourceRangeMergeIterator = sparse_container::parallel_iterator<ResourceAccessRangeMap, const ResourceAccessRangeMap>;

// Range generators for to allow event scope filtration to be limited to the top of the resource access traversal pipeline
//
// Note: there is no "begin/end" or reset facility.  These are each written as "one time through" generators.
//
// Usage:
//  Constructor() -- initializes the generator to point to the begin of the space declared.
//  *  -- the current range of the generator empty signfies end
//  ++ -- advance to the next non-empty range (or end)

// Generate the ranges that are the intersection of range and the entries in the RangeMap
template <typename RangeMap, typename KeyType = typename RangeMap::key_type>
class MapRangesRangeGenerator {
  public:
    using key_type = KeyType;
    // Default constructed is safe to dereference for "empty" test, but for no other operation.
    MapRangesRangeGenerator(const RangeMap &map, const KeyType &range) : gen_(MapGen(map), RangeGen(range)) {}
    MapRangesRangeGenerator(const MapRangesRangeGenerator &from) = default;

    const KeyType &operator*() const { return *gen_; }
    const KeyType *operator->() const { return &(*gen_); }
    MapRangesRangeGenerator &operator++() {
        ++gen_;
        return *this;
    }
    const key_type &seek_forward(const key_type &range) { return gen_.seek_forward(range); }

  protected:
    using MapGen = sparse_container::mapped_range_generator<RangeMap>;
    using RangeGen = sparse_container::single_range_generator<KeyType>;
    using IntersectionGen = sparse_container::intersection_generator<MapGen, RangeGen>;
    IntersectionGen gen_;
};

// Generate the ranges that are the intersection of the RangeGen ranges and the entries in the FilterMap
// Templated to allow for different Range generators or map sources...
template <typename RangeMap, typename RangeGen, typename KeyType = typename RangeMap::key_type>
class FilteredGeneratorGenerator {
  public:
    using key_type = KeyType;
    FilteredGeneratorGenerator(const RangeMap &map, const RangeGen &gen) : gen_(MapGen(map), gen) {}
    FilteredGeneratorGenerator(const FilteredGeneratorGenerator &from) = default;
    const KeyType &operator*() const { return *gen_; }
    const KeyType *operator->() const { return &(*gen_); }
    FilteredGeneratorGenerator &operator++() {
        ++gen_;
        return *this;
    }
    const key_type &seek_forward(const key_type &range) { return gen_.seek_forward(range); }

  protected:
    using MapGen = sparse_container::mapped_range_generator<RangeMap>;
    using IntersectionGen = sparse_container::intersection_generator<MapGen, RangeGen>;
    IntersectionGen gen_;
};

using EventImageRangeGenerator = FilteredGeneratorGenerator<ResourceAccessRangeMap, ImageRangeGen>;
using EventSimpleRangeGenerator = MapRangesRangeGenerator<ResourceAccessRangeMap>;

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
        // const AccessStateConstPtr &DefaultState() const;
        OpID Op() const;
        OpContext(const AccessContext &access_context);
        OpContext(const OpContext &other_context);

      protected:
        static std::atomic<OpID> next_op_;  // Ops must be unique over the life of a device (at least)

        const AccessContext &context_;
        const OpID op_id_;
    };

    using StateConstPtr = std::shared_ptr<const ResourceAccessState>;
    using PendingConstPtr = std::shared_ptr<const PendingBarriers>;
    using FirstConstPtr = std::shared_ptr<const FirstAccesses>;

    using StatePtr = std::shared_ptr<ResourceAccessState>;
    using PendingPtr = std::shared_ptr<PendingBarriers>;
    using FirstPtr = std::shared_ptr<FirstAccesses>;

    static const StateConstPtr &DefaultResourceState() { return default_resource_state_; }
    static const PendingConstPtr &DefaultPendingState() { return default_pending_state_; }
    static const FirstConstPtr &DefaultFirstState() { return default_first_state_; }
    static const AccessStateConstPtr &DefaultAccessState() { return default_access_state_; }

    static const StateConstPtr default_resource_state_;
    static const PendingConstPtr default_pending_state_;
    static const FirstConstPtr default_first_state_;
    static const FirstConstPtr default_first_state_closed_;
    static const AccessStateConstPtr default_access_state_;

    AccessState() = delete;
    AccessState(const AccessState &other) = default;

    // Constructors for SSA State Updates
    //     ... that update state and first
    AccessState(StatePtr &&state, FirstPtr &&first) : state_(std::move(state)), pending_(), first_(std::move(first)) {}
    //     ... that update state but first is closed or unaffected
    AccessState(const AccessState &other, StatePtr &&state) : state_(std::move(state)), pending_(), first_(other.first_) {}
    //     ... that only update the pending barriers
    AccessState(const AccessState &other, PendingPtr &&pending)
        : state_(other.state_), pending_(std::move(pending)), first_(other.first_) {}
    //     ... for submitted states
    AccessState(StatePtr &&state, const FirstAccesses::ForceClosed &)
        : state_(std::move(state)), pending_(), first_(default_first_state_closed_) {}

    // General Constructor
    AccessState(const StateConstPtr &state, const PendingConstPtr &pending, const FirstConstPtr &first)
        : state_(state), pending_(pending), first_(first) {}

    const OpResult &GetResult() const { return result_; }
    const ResourceAccessState &GetState() const { return *state_; }
    const FirstAccesses &GetFirstAccess() const { return *first_; }
    const PendingBarriers &GetPending() const { return *pending_; }

    // NOTE: Updates mutable member, though const
    void UpdateResult(OpResult &&from) const { result_ = std::move(from); }

    template <typename Updater>
    static void SSAUpdate(const OpContext &op, const Updater &updater, AccessStateConstPtr &access);

    static AccessStateConstPtr SSAResolveFirsts(const OpContext &op, const AccessStateConstPtr &access_state_ptr_a,
                                                const AccessStateConstPtr &access_state_ptr_b, StatePtr &&update_state);
    static AccessStateConstPtr SSAResolve(const OpContext &op, const AccessStateConstPtr &lhs, const AccessStateConstPtr &rhs);

    static AccessStateConstPtr SSAUpdateAccess(const OpContext &op_context, const AccessStateConstPtr &from,
                                               const SyncStageAccessInfoType &usage_info, const SyncOrdering ordering_rule,
                                               const ResourceUsageTag tag);

    template <typename Updater>
    static AccessStateConstPtr SSAUpdateStateAndFirst(const OpContext &op_context, const AccessStateConstPtr &from,
                                                      const Updater &updater);
    template <typename Scope>
    static AccessStateConstPtr SSAApplyBarrierStack(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                    const SyncBarrierStack &barrier_stack);
    static AccessStateConstPtr SSAApplyPendingBarriers(const OpContext &op_context, const AccessState &from, ResourceUsageTag tag);

    static AccessStateConstPtr SSAApplyPendingBarriers(const OpContext &op_context, const AccessState &from, ResourceUsageTag tag,
                                                       const PendingBarriers &pending);
    template <typename Scope, typename BarrierIt>
    static AccessStateConstPtr SSAApplyScopedBarriers(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                      const BarrierIt begin, const BarrierIt end, bool layout_transition,
                                                      ResourceUsageTag tag);
    template <typename Scope, typename BarrierIt>
    static AccessStateConstPtr SSAUpdateScopedBarriers(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                       const BarrierIt begin, const BarrierIt end, bool layout_transition);

    static AccessStateConstPtr SSAApplySemaphoreBarriers(const OpContext &op_context, const AccessState &from,
                                                         const SemaphoreScope &signal, const SemaphoreScope wait);

    static AccessStateConstPtr SSANormalizeAndClearFirstUse(const OpContext &op_context, const AccessState &from);
    template <typename CullingTransform>
    static AccessStateConstPtr SSACullingUpdateState(const OpContext &op_context, const AccessState &from,
                                                     const CullingTransform &transform);

  private:
    // Treat the access state as a "static single assignment" of a ResourceAccessState, allowing safe sharing by many address
    // ranges, with the results of a given op
    // TODO: Make this const when we have constructors that can apply update operations during construction
    std::shared_ptr<const ResourceAccessState> state_;
    std::shared_ptr<const PendingBarriers> pending_;
    std::shared_ptr<const FirstAccesses> first_;

    // Temporary result for current operation on AccessContext
    mutable OpResult result_;
};

using RangeGenWrapper = sparse_container::range_gen_wrapper<ResourceAccessRange>;
using MapGapGen = sparse_container::mapped_gap_generator<ResourceAccessRangeMap>;
using MapRangeGen = sparse_container::mapped_range_generator<ResourceAccessRangeMap>;
using IntersectionAdvanceSecond = sparse_container::intersection_advance_policy_second;

struct ApplyTrackbackStackAction {
    explicit ApplyTrackbackStackAction(const std::vector<SyncBarrier> &barriers_,
                                       const ApplyTrackbackStackAction *previous_barrier_ = nullptr)
        : barriers(barriers_), previous_barrier(previous_barrier_) {}
    void operator()(ResourceAccessState &access) const {
        // Layout transitions are *not* supported
        access.ApplyBarriersImmediate(barriers);
        if (previous_barrier) {
            (*previous_barrier)(access);
        }
    }
    const std::vector<SyncBarrier> &barriers;
    const ApplyTrackbackStackAction *previous_barrier;
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
    using ImageRangeGenerator = subresource_adapter::ImageRangeGenerator;
    enum DetectOptions : uint32_t {
        kDetectPrevious = 1U << 0,
        kDetectAsync = 1U << 1,
        kDetectAll = (kDetectPrevious | kDetectAsync)
    };

    struct Ops {
        struct UpdateAccess {
            const SyncStageAccessInfoType &usage_info;
            const SyncOrdering ordering_rule;
            const ResourceUsageTag tag;

            UpdateAccess(SyncStageAccessIndex usage_, SyncOrdering ordering_rule_, ResourceUsageTag tag_)
                : usage_info(SyncStageAccess::UsageInfo(usage_)), ordering_rule(ordering_rule_), tag(tag_) {}

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct OffsetTags {
            const ResourceUsageTag offset;

            void Update(ResourceAccessState &state) const { state.OffsetTag(offset); };
            void Update(FirstAccesses &first) const { first.OffsetTag(offset); }

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct ApplyAcquireNextSemaphore {
            ApplyAcquireNextSemaphore(const SyncExecScope &wait_scope, ResourceUsageTag acquire_tag_);

            const SyncBarrier barrier;
            const ResourceUsageTag acquire_tag;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;

          private:
            // kPresentSrcScope/kPresentValidAccesses cannot be regular global variables, because they use global
            // variables from another compilation unit (through syncStageAccessMaskByStageBit() call) for initialization,
            // and initialization of globals between compilation units is undefined. Instead they get initialized
            // on the first use (it's important to ensure this first use is also not initialization of some global!).
            const SyncExecScope &getPresentSrcScope() const;
            const SyncStageAccessFlags &getPresentValidAccesses() const;
        };

        struct NormalizeAndClearFirstUse {
            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct ApplyQueueScopeBarrier {
            const SyncBarrier &barrier;
            const ResourceAccessState::QueueScopeOps scope;
            const ResourceUsageTag tag;
            const bool layout_transition;

            static bool InitLayoutTransition(QueueId queue_id, bool layout_transition);

            ApplyQueueScopeBarrier(QueueId queue_id, const SyncBarrier &barrier_, bool layout_transition_ = false,
                                   ResourceUsageTag tag_ = kInvalidTag);

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct ApplyRenderPassLayoutTransition {
            const std::vector<SyncBarrier> &barriers;
            const ResourceAccessState::UntaggedScopeOps scope;
            const ResourceUsageTag tag;

            ApplyRenderPassLayoutTransition(const std::vector<SyncBarrier> &barriers_, ResourceUsageTag tag_);

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct ApplyPendingBarriers {
            ResourceUsageTag tag;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct ApplySemaphoreBarrier {
            const SemaphoreScope signal;
            const SemaphoreScope wait;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct SetQueueOffsetTags {
            const QueueId queue;
            const ResourceUsageTag offset;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;

            void Update(ResourceAccessState &state) const;
            ;
            void Update(FirstAccesses &first) const;
        };

        struct UpdatePendingEventBarriers {
            ResourceAccessState::EventScopeOps scope;
            SyncBarrier barrier;
            bool layout_transition;  // Implies Infill default

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct UpdatePendingGlobalEventBarriers {
            // Does not apply barriers, intentionally
            ResourceAccessState::EventScopeOps scope;
            const std::vector<SyncBarrier> &barriers;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct UpdatePendingPipelineBarriers {
            ResourceAccessState::QueueScopeOps scope;
            SyncBarrier barrier;
            bool layout_transition;  // Implies Infill default

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };

        struct UpdateAndApplyGlobalPipelineBarriers {
            ResourceAccessState::QueueScopeOps scope;
            ResourceUsageTag tag;
            const std::vector<SyncBarrier> &barriers;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };
        struct ApplyBarrierStack {
            const ResourceAccessState::UntaggedScopeOps scope;
            const SyncBarrierStack barrier_stack;

            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
            ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const ApplyBarrierStack *prev);
            ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const bool layout_transition, const ResourceUsageTag tag,
                              const ApplyBarrierStack &prev);
            ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const bool layout_transition, const ResourceUsageTag tag,
                              const SyncBarrierStack *prev = nullptr);
        };
        template <typename CullingTransform>
        struct ApplyCullingUpdateState {
            const CullingTransform &transform;
            AccessStateConstPtr Update(const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const;
        };
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

    void ResolveFromContext(const AccessContext &from_context, const Ops::OffsetTags &offset_tags_op);

    void ResolveFromContext(const AccessContext &from_context, const Ops::SetQueueOffsetTags &);

    void ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen,
                            const Ops::ApplyAcquireNextSemaphore &op);

    void ResolveFromContext(const AccessContext &from_context, const Ops::ApplyQueueScopeBarrier &barrier_op);
    void ResolveFromContext(const AccessContext &from_context, const Ops::ApplyBarrierStack &barrier_op);
    void ResolveFromContext(const AccessContext &from_context, const ImageRangeGenerator &range_gen,
                            const Ops::ApplyQueueScopeBarrier &barrier_op);
    void ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen,
                            const Ops::ApplyRenderPassLayoutTransition &);

    void ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen, const Ops::ApplySemaphoreBarrier &);
    void ResolveFromContext(const AccessContext &from_context, const Ops::ApplySemaphoreBarrier &);

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

    AccessContext(uint32_t subpass, VkQueueFlags queue_flags, const std::vector<SubpassDependencyGraphNode> &dependencies,
                  const std::vector<AccessContext> &contexts, const AccessContext *external_context);

    AccessContext() { Reset(); }
    AccessContext(const AccessContext &copy_from) = default;
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

    void UpdateMemoryAccessState(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                                 const Ops::ApplyRenderPassLayoutTransition &transition_op);

    void UpdateMemoryAccessState(const SingleRangeGen &range_gen, const Ops::UpdatePendingPipelineBarriers &);
    void UpdateMemoryAccessState(const ImageRangeGen &range_gen, const Ops::UpdatePendingPipelineBarriers &);
    void UpdateMemoryAccessState(const SingleRangeGen &range_gen, const Ops::UpdateAndApplyGlobalPipelineBarriers &);

    void UpdateMemoryAccessState(const EventSimpleRangeGenerator &range_gen, const Ops::UpdatePendingGlobalEventBarriers &);
    void UpdateMemoryAccessState(const EventImageRangeGenerator &range_gen, const Ops::UpdatePendingEventBarriers &);
    void UpdateMemoryAccessState(const EventSimpleRangeGenerator &range_gen, const Ops::UpdatePendingEventBarriers &);
    void UpdateMemoryAccessState(const Ops::ApplyPendingBarriers &);

  private:
    template <typename Action, typename RangeGen>
    void UpdateMemoryAccessState(const Action &action, const RangeGen &range_gen, bool infill_default);

    template <typename ResolveOp>
    void ResolveFromContextImpl(ResolveOp &&resolve_op, const AccessContext &from_context, bool infill_default,
                                bool recur_to_infill);
    template <typename ResolveOp, typename RangeGenerator>
    void ResolveFromContextImpl(const ResolveOp &resolve_op, const RangeGenerator &range_gen, const AccessContext &from_context,
                                bool infill_default, bool recur_to_infill);
    template <typename RangeGen>
    static ResourceAccessRangeMap InfillRangeGen(const RangeGen &gen_arg, const ResourceAccessStateFunction *previous_barrier);
    template <typename Updater>
    void ForAll(Updater &&updater);

    template <typename Action>
    void ConstForAll(Action &&action) const;

    AccessState::OpContext MakeOpContext() const { return AccessState::OpContext(*this); }
    template <typename Action>
    void UpdateMemoryAccessRangeState(Action &action, const ResourceAccessRange &range, bool infill_default);
    template <typename Predicate>
    void ApplyPredicatedWait(const Predicate &predicate);

    // Follow the context previous to access the access state, supporting "lazy" import into the context. Not intended for
    // subpass layout transition, as the pending state handling is more complex
    // TODO: See if returning the lower_bound would be useful from a performance POV -- look at the lower_bound overhead
    // Would need to add a "hint" overload to parallel_iterator::invalidate_[AB] call, if so.
    // NOTE: The use of the wrapper prevents recursive template expansion.
    ResourceAccessRangeMap ResolvePreviousAccessRangeGen(const RangeGenWrapper &gen_arg, const AccessState::OpContext &op,
                                                         const Ops::ApplyBarrierStack *previous_barrier = nullptr) const;

    template <typename RangeGen, typename BarrierAction>
    ResourceAccessRangeMap ResolveAccessRangeGenNoRecur(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                        const AccessState::OpContext &op) const;

    template <typename RangeGen, typename BarrierAction>
    ResourceAccessRangeMap ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                 const AccessState::OpContext &op, bool recur_to_infill) const;
    template <typename RangeGen>
    ResourceAccessRangeMap ResolveAccessRangeGen(const RangeGen &gen_arg, Ops::ApplyBarrierStack &barrier_action,
                                                 const AccessState::OpContext &op, bool recur_to_infill) const;
    template <typename Detector>
    HazardResult DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const;
    template <typename Detector, typename RangeGen>
    HazardResult DetectHazardGeneratedRanges(Detector &detector, RangeGen &range_gen, DetectOptions options) const;
    template <typename RangeGen>
    void InfillGapRangeGen(const RangeGen &gen_arg, const AccessStateConstPtr &infill_state);
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
