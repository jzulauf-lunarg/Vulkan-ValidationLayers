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

#include <cinttypes>
#include "state_tracker/buffer_state.h"
#include "sync/sync_access_context.h"

std::atomic<AccessState::OpID> AccessState::OpContext::next_op_(0);

const AccessState::StateConstPtr AccessState::default_resource_state_{std::make_shared<const ResourceAccessState>()};
const AccessState::PendingConstPtr AccessState::default_pending_state_;
const AccessState::FirstConstPtr AccessState::default_first_state_{std::make_shared<const FirstAccesses>()};
const AccessState::FirstConstPtr AccessState::default_first_state_closed_{
    std::make_shared<const FirstAccesses>(FirstAccesses::ForceClosed())};
const AccessStateConstPtr AccessState::default_access_state_{std::make_shared<const AccessState>(
    AccessState::default_resource_state_, AccessState::default_pending_state_, AccessState::default_first_state_)};

size_t AccessStateStats::total_entries = 0;
size_t AccessStateStats::reused_entries = 0;
std::array<size_t, static_cast<size_t>(MismatchReason::COUNT)> AccessStateStats::total_compare_stats;

bool SimpleBinding(const BINDABLE &bindable) { return !bindable.sparse && bindable.Binding(); }
VkDeviceSize ResourceBaseAddress(const vvl::Buffer &buffer) { return buffer.GetFakeBaseAddress(); }

class HazardDetector {
    const SyncStageAccessInfoType &usage_info_;

  public:
    HazardResult Detect(const AccessStateConstPtr &state_ptr) const { return state_ptr->GetState().DetectHazard(usage_info_); }
    HazardResult DetectAsync(const AccessStateConstPtr &state_ptr, ResourceUsageTag start_tag) const {
        return state_ptr->GetState().DetectAsyncHazard(usage_info_, start_tag);
    }
    explicit HazardDetector(SyncStageAccessIndex usage_index) : usage_info_(SyncStageAccess::UsageInfo(usage_index)) {}
};

class HazardDetectorWithOrdering {
    const SyncStageAccessInfoType &usage_info_;
    const SyncOrdering ordering_rule_;

  public:
    HazardResult Detect(const AccessStateConstPtr &state_ptr) const {
        return state_ptr->GetState().DetectHazard(usage_info_, ordering_rule_, kQueueIdInvalid);
    }
    HazardResult DetectAsync(const AccessStateConstPtr &state_ptr, ResourceUsageTag start_tag) const {
        return state_ptr->GetState().DetectAsyncHazard(usage_info_, start_tag);
    }
    HazardDetectorWithOrdering(SyncStageAccessIndex usage_index, SyncOrdering ordering)
        : usage_info_(SyncStageAccess::UsageInfo(usage_index)), ordering_rule_(ordering) {}
};

class HazardDetectFirstUse {
  public:
    HazardDetectFirstUse(const AccessStateConstPtr &recorded_use, QueueId queue_id, const ResourceUsageRange &tag_range)
        : recorded_use_(recorded_use), queue_id_(queue_id), tag_range_(tag_range) {}
    HazardResult Detect(const AccessStateConstPtr &state_ptr) const {
        return state_ptr->GetState().DetectHazard(recorded_use_->GetFirstAccess(), queue_id_, tag_range_);
    }
    HazardResult DetectAsync(const AccessStateConstPtr &state_ptr, ResourceUsageTag start_tag) const {
        return state_ptr->GetState().DetectAsyncHazard(recorded_use_->GetFirstAccess(), tag_range_, start_tag);
    }

  private:
    const AccessStateConstPtr recorded_use_;
    const QueueId queue_id_;
    const ResourceUsageRange &tag_range_;
};

AccessContext::AccessContext(uint32_t subpass, VkQueueFlags queue_flags,
                             const std::vector<SubpassDependencyGraphNode> &dependencies,
                             const std::vector<AccessContext> &contexts, const AccessContext *external_context) {
    Reset();
    const auto &subpass_dep = dependencies[subpass];
    const bool has_barrier_from_external = subpass_dep.barrier_from_external.size() > 0U;
    prev_.reserve(subpass_dep.prev.size() + (has_barrier_from_external ? 1U : 0U));
    prev_by_subpass_.resize(subpass, nullptr);  // Can't be more prevs than the subpass we're on
    for (const auto &prev_dep : subpass_dep.prev) {
        const auto prev_pass = prev_dep.first->pass;
        const auto &prev_barriers = prev_dep.second;
        assert(prev_dep.second.size());
        prev_.emplace_back(&contexts[prev_pass], queue_flags, prev_barriers);
        prev_by_subpass_[prev_pass] = &prev_.back();
    }

    async_.reserve(subpass_dep.async.size());
    for (const auto async_subpass : subpass_dep.async) {
        // Start tags are not known at creation time (as it's done at BeginRenderpass)
        async_.emplace_back(contexts[async_subpass], kInvalidTag);
    }

    if (has_barrier_from_external) {
        // Store the barrier from external with the reat, but save pointer for "by subpass" lookups.
        prev_.emplace_back(external_context, queue_flags, subpass_dep.barrier_from_external);
        src_external_ = &prev_.back();
    }
    if (subpass_dep.barrier_to_external.size()) {
        dst_external_ = TrackBack(this, queue_flags, subpass_dep.barrier_to_external);
    }
}

// Note that second argument is modified in place.
template <typename Updater>
void AccessState::SSAUpdate(const OpContext &op, const Updater &updater, AccessStateConstPtr &map_entry) {
#if 0
    // WIP WIP WIP
    if (!map_entry) {
        assert(map_entry);
    }
#endif
    if (op.Op() == map_entry->result_.op) {
        // overwrite the map entry with the result
        map_entry = map_entry->result_.state;
    } else {
        AccessStateConstPtr update(updater.Update(op, map_entry));
        if (update) {
            // overwrite the map entry with the result
            map_entry = std::move(update);
        }
    }

#if 0
    // WIP WIP WIP
    if (!map_entry) {
        assert(map_entry);
    }
#endif
}

// Resolving the firsts after we resolve the state, allows us to finalize the SSA new state creation
AccessStateConstPtr AccessState::SSAResolveFirsts(const OpContext &op, const AccessStateConstPtr &access_state_ptr_a,
                                                  const AccessStateConstPtr &access_state_ptr_b, StatePtr &&update_state) {
    const FirstAccesses *first_a = access_state_ptr_a->first_.get();
    const FirstAccesses *first_b = access_state_ptr_b->first_.get();
    if (first_a) {
        if (first_a->closed || !first_b) {
            return std::make_shared<const AccessState>(*access_state_ptr_a, std::move(update_state));
        }
        // Note: Neither a nor b are null
    } else if (first_b) {
        if (first_b->closed || !first_a) {
            return std::make_shared<const AccessState>(*access_state_ptr_b, std::move(update_state));
        }
        // Note: Neither a nor b are null
    } else {
        // Note: both a and b are null
        return std::make_shared<const AccessState>(std::move(update_state), FirstPtr());
    }

    // We only fall through if both a and b firsts are non-null
    FirstPtr update_first = std::make_shared<FirstAccesses>(*first_a);
    update_first->Merge(*first_b);
    return std::make_shared<const AccessState>(std::move(update_state), std::move(update_first));
}

// NOTE: If the maked_shared copies of other become a hotspot, we can implement a policy (template) parameter as to
//       shallow or deep copies w.r.t. the AccessStates are in the same AccessContext list.
//       That policy would be specified in the calling context based on whether the AccessStates being merged were
//       created with insert_points in the same or different AccessContexts, but we don't want to store this on a per
//       AccessState basis.
//
// The logic behind resolves is the same as update, we assume that earlier hazards have be reported, and that no
// tranistive hazard can exists with a hazard between the earlier operations.  Yes, an early hazard can mask that another
// exists, but if you fix *that* hazard it either fixes or unmasks the subsequent ones.
AccessStateConstPtr AccessState::SSAResolve(const OpContext &op, const AccessStateConstPtr &access_state_ptr_a,
                                            const AccessStateConstPtr &access_state_ptr_b) {
    AccessStateConstPtr resolved;

    // Set if we have resolved this before... (noting that result is mutable)
    OpResult &result_a = access_state_ptr_a->result_;
    OpResult &result_b = access_state_ptr_b->result_;
    if (result_a.op == op.Op()) {
        resolved = result_a.LookupResolve(access_state_ptr_b);
        if (resolved) return resolved;
    } else if (result_b.op == op.Op()) {
        resolved = result_b.LookupResolve(access_state_ptr_a);
        if (resolved) return resolved;
    }
    const ResourceAccessState &state_a = *access_state_ptr_a->state_;
    const ResourceAccessState &state_b = *access_state_ptr_b->state_;
    if (state_a.HasWriteOp()) {
        if (state_b.HasWriteOp()) {
            const ResourceUsageTag write_tag_a = state_a.LastWriteTag();
            const ResourceUsageTag write_tag_b = state_b.LastWriteTag();
            // NOTE: Both a and b have writes, and thus first access is "closed". If they differ, we are selecting the latter's
            //       first_ state, but it and this can only differ if there are async hazards errors,  and we've reported any
            //       exsiting hazard, and we can just overwrite with the more recent operation
            if (write_tag_a < write_tag_b) {
                resolved = access_state_ptr_b;
            } else if (write_tag_b < write_tag_a) {
                resolved = access_state_ptr_a;
            } else {
                // In the *equals* case for write operations, we merged the write barriers and the read state (but without the
                // dependency chaining logic or any stage expansion)
                StatePtr update_state = std::make_shared<ResourceAccessState>(state_a);
                update_state->ResolveReadWrite(state_b);
                resolved = SSAResolveFirsts(op, access_state_ptr_a, access_state_ptr_b, std::move(update_state));
            }
        } else {
            // a has a write and other doesn't -- at best async read in other, which have been reported, and will be dropped
            // Since a has a write first access is closed and shouldn't be updated by other
            resolved = access_state_ptr_a;
        }
    } else if (state_b.HasWriteOp()) {
        // b has write a doesn't, thus choose b, See first access NOTE above
        resolved = access_state_ptr_b;
    } else {  // not this->last_write OR other.last_write
        // Neither state has a write, just merge the reads and the firsts
        StatePtr update_state = std::make_shared<ResourceAccessState>(state_a);
        update_state->ResolveRead(state_b);
        resolved = SSAResolveFirsts(op, access_state_ptr_a, access_state_ptr_b, std::move(update_state));
    }

    // Add anything new to the cache
    if (resolved) {
        // We look up both ways, so we can just add one way.
        if (result_a.op != op.Op()) {
            result_a.InitResolve(op.Op());
        }
        result_a.CacheResolve(access_state_ptr_b, resolved);
    }

    return resolved;
}

template <typename NormalizeOp>
void AccessContext::Trim(NormalizeOp &&normalize) {
    ForAll(normalize);
    sparse_container::consolidate(access_state_map_);
}

void AccessContext::TrimAndClearFirstAccess() { Trim(Ops::NormalizeAndClearFirstUse()); }

void AccessContext::AddReferencedTags(ResourceUsageTagSet &used) const {
    auto gather = [&used](const ResourceAccessRangeMap::value_type &access) {
        access.second->GetState().GatherReferencedTags(used);
    };
    ConstForAll(gather);
}

template <typename Updater>
static void ForAllRangeMap(Updater &&updater, AccessState::OpContext op_context, ResourceAccessRangeMap &map) {
    for (auto &access : map) {
        AccessState::SSAUpdate(op_context, updater, access.second);
    }
}

template <typename Updater>
void AccessContext::ForAll(Updater &&updater) {
    AccessState::OpContext op_context = MakeOpContext();
    ForAllRangeMap(updater, op_context, access_state_map_);
}

template <typename Action>
void AccessContext::ConstForAll(Action &&action) const {
    for (auto &access : access_state_map_) {
        action(access);
    }
}

template <typename RangeGen>
void AccessContext::InfillGapRangeGen(const RangeGen &gen_arg, const AccessStateConstPtr &infill_state) {
    sparse_container::mapped_gap_generator gap_gen(access_state_map_);
    sparse_container::intersection_generator infill_gen(gen_arg, gap_gen);

    for (; infill_gen->non_empty(); ++infill_gen) {
        // Inserting before the gap in access_state_map_, the won't confuse the generator when it advances, as
        // pos is after the current_range in the generator, and insertion doesn't invalidate iterators
        access_state_map_.insert(infill_gen.get_second_gen().get_pos(), std::make_pair(*infill_gen, infill_state));
    }
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen,
                                       const Ops::ApplyRenderPassLayoutTransition &transition_op) {
    // Layout transition must infill with default to set the layout transition correctly
    ResolveFromContextImpl(transition_op, range_gen, from_context, true, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen,
                                       const Ops::ApplySemaphoreBarrier &barrier_op) {
    ResolveFromContextImpl(barrier_op, range_gen, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const Ops::ApplySemaphoreBarrier &barrier_op) {
    ResolveFromContextImpl(barrier_op, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from) {
    ResolveAccessMaps(MakeOpContext(), from.access_state_map_, access_state_map_);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const Ops::OffsetTags &offset_tags_op) {
    ResolveFromContextImpl(offset_tags_op, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const Ops::SetQueueOffsetTags &queue_tag_op) {
    ResolveFromContextImpl(queue_tag_op, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const ImageRangeGen &range_gen,
                                       const Ops::ApplyAcquireNextSemaphore &acquire_op) {
    ResolveFromContextImpl(acquire_op, range_gen, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const Ops::ApplyQueueScopeBarrier &barrier_op) {
    ResolveFromContextImpl(barrier_op, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const Ops::ApplyBarrierStack &barrier_op) {
    ResolveFromContextImpl(barrier_op, from_context, false, false);
}

void AccessContext::ResolveFromContext(const AccessContext &from_context, const ImageRangeGenerator &range_gen,
                                       const Ops::ApplyQueueScopeBarrier &barrier_op) {
    ResolveFromContextImpl(barrier_op, range_gen, from_context, false, false);
}

// Non-lazy import of all accesses, WaitEvents needs this.
void AccessContext::ResolvePreviousAccesses() {
    if (!prev_.size()) return;  // If no previous contexts, nothing to do

    auto prev_op = MakeOpContext();
    SingleRangeGen single_gen(kFullRange);
    RangeGenWrapper full_range_gen(single_gen);
    ResolveAccessMaps(MakeOpContext(), ResolvePreviousAccessRangeGen(full_range_gen, prev_op, nullptr), access_state_map_);
    InfillGapRangeGen(single_gen, AccessState::DefaultAccessState());
}

void AccessContext::UpdateAccessState(const vvl::Buffer &buffer, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                                      const ResourceAccessRange &range, const ResourceUsageTag tag) {
    if (!SimpleBinding(buffer)) return;
    const auto base_address = ResourceBaseAddress(buffer);
    Ops::UpdateAccess action(current_usage, ordering_rule, tag);
    UpdateMemoryAccessRangeState(action, range + base_address, true);
}

void AccessContext::UpdateAccessState(const ImageState &image, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                                      const VkImageSubresourceRange &subresource_range, const ResourceUsageTag &tag) {
    // range_gen is non-temporary to avoid an additional copy
    ImageRangeGen range_gen = image.MakeImageRangeGen(subresource_range, false);
    UpdateAccessState(range_gen, current_usage, ordering_rule, tag);
}
void AccessContext::UpdateAccessState(const ImageState &image, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                                      const VkImageSubresourceRange &subresource_range, const VkOffset3D &offset,
                                      const VkExtent3D &extent, const ResourceUsageTag tag) {
    // range_gen is non-temporary to avoid an additional copy
    ImageRangeGen range_gen = image.MakeImageRangeGen(subresource_range, offset, extent, false);
    UpdateAccessState(range_gen, current_usage, ordering_rule, tag);
}

void AccessContext::UpdateAccessState(const ImageViewState &image_view, SyncStageAccessIndex current_usage,
                                      SyncOrdering ordering_rule, const VkOffset3D &offset, const VkExtent3D &extent,
                                      const ResourceUsageTag tag) {
    // range_gen is non-temporary to avoid an additional copy
    ImageRangeGen range_gen(image_view.MakeImageRangeGen(offset, extent));
    UpdateAccessState(range_gen, current_usage, ordering_rule, tag);
}

void AccessContext::UpdateAccessState(const ImageViewState &image_view, SyncStageAccessIndex current_usage,
                                      SyncOrdering ordering_rule, ResourceUsageTag tag) {
    // Get is const, and will be copied in callee
    UpdateAccessState(image_view.GetFullViewImageRangeGen(), current_usage, ordering_rule, tag);
}

void AccessContext::UpdateAccessState(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                                      SyncStageAccessIndex current_usage, SyncOrdering ordering_rule, const ResourceUsageTag tag) {
    const std::optional<ImageRangeGen> &attachment_gen = view_gen.GetRangeGen(gen_type);
    if (attachment_gen) {
        // Value of const optional is const, and will be copied in callee
        UpdateAccessState(*attachment_gen, current_usage, ordering_rule, tag);
    }
}

void AccessContext::UpdateAccessState(ImageRangeGen &range_gen, SyncStageAccessIndex current_usage, SyncOrdering ordering_rule,
                                      ResourceUsageTag tag) {
    Ops::UpdateAccess action(current_usage, ordering_rule, tag);
    UpdateMemoryAccessState(action, range_gen, true);
}

void AccessContext::UpdateAccessState(const ImageRangeGen &range_gen, SyncStageAccessIndex current_usage,
                                      SyncOrdering ordering_rule, ResourceUsageTag tag) {
    // range_gen is non-temporary to avoid infinite call recursion
    ImageRangeGen mutable_range_gen(range_gen);
    UpdateAccessState(mutable_range_gen, current_usage, ordering_rule, tag);
}

void AccessContext::UpdateMemoryAccessState(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                                            const Ops::ApplyRenderPassLayoutTransition &transition_op) {
    const std::optional<ImageRangeGen> &attachment_gen = view_gen.GetRangeGen(gen_type);
    if (attachment_gen) {
        UpdateMemoryAccessState(transition_op, *attachment_gen, true);
    }
}

void AccessContext::UpdateMemoryAccessState(const SingleRangeGen &range_gen, const Ops::UpdatePendingPipelineBarriers &barrier_op) {
    // Only infill on layout transitions
    UpdateMemoryAccessState(barrier_op, range_gen, barrier_op.layout_transition);
}

void AccessContext::UpdateMemoryAccessState(const ImageRangeGen &range_gen, const Ops::UpdatePendingPipelineBarriers &barrier_op) {
    UpdateMemoryAccessState(barrier_op, range_gen, barrier_op.layout_transition);
}

void AccessContext::UpdateMemoryAccessState(const SingleRangeGen &range_gen,
                                            const Ops::UpdateAndApplyGlobalPipelineBarriers &barrier_op) {
    UpdateMemoryAccessState(barrier_op, range_gen, false);
}

void AccessContext::UpdateMemoryAccessState(const EventSimpleRangeGenerator &range_gen,
                                            const Ops::UpdatePendingGlobalEventBarriers &barrier_op) {
    UpdateMemoryAccessState(barrier_op, range_gen, false);
}
void AccessContext::UpdateMemoryAccessState(const EventImageRangeGenerator &range_gen,
                                            const Ops::UpdatePendingEventBarriers &barrier_op) {
    UpdateMemoryAccessState(barrier_op, range_gen, barrier_op.layout_transition);
}
void AccessContext::UpdateMemoryAccessState(const EventSimpleRangeGenerator &range_gen,
                                            const Ops::UpdatePendingEventBarriers &barrier_op) {
    UpdateMemoryAccessState(barrier_op, range_gen, barrier_op.layout_transition);
}

void AccessContext::UpdateMemoryAccessState(const Ops::ApplyPendingBarriers &apply_pending_op) { ForAll(apply_pending_op); }

void AccessContext::ResolveChildContexts(const std::vector<AccessContext> &contexts) {
    for (uint32_t subpass_index = 0; subpass_index < contexts.size(); subpass_index++) {
        auto &context = contexts[subpass_index];
        Ops::ApplyBarrierStack barrier_action(context.GetDstExternalTrackBack().barriers, false, kInvalidTag);
        ResolveFromContext(context, barrier_action);
    }
}

// Caller must ensure that lifespan of this is less than the lifespan of from
void AccessContext::ImportAsyncContexts(const AccessContext &from) {
    async_.insert(async_.end(), from.async_.begin(), from.async_.end());
}

// Suitable only for *subpass* access contexts
HazardResult AccessContext::DetectSubpassTransitionHazard(const TrackBack &track_back, const AttachmentViewGen &attach_view) const {
    if (!attach_view.IsValid()) return HazardResult();

    // We should never ask for a transition from a context we don't have
    assert(track_back.source_subpass);

    // Do the detection against the specific prior context independent of other contexts.  (Synchronous only)
    // Hazard detection for the transition can be against the merged of the barriers (it only uses src_...)
    const SyncBarrier merged_barrier(track_back.barriers);
    HazardResult hazard = track_back.source_subpass->DetectImageBarrierHazard(attach_view, merged_barrier, kDetectPrevious);
    if (!hazard.IsHazard()) {
        // The Async hazard check is against the current context's async set.
        SyncBarrier null_barrier = {};
        hazard = DetectImageBarrierHazard(attach_view, null_barrier, kDetectAsync);
    }

    return hazard;
}

void AccessContext::AddAsyncContext(const AccessContext *context, ResourceUsageTag tag) {
    if (context) {
        async_.emplace_back(*context, tag);
    }
}

HazardResult AccessContext::DetectHazard(const vvl::Buffer &buffer, SyncStageAccessIndex usage_index,
                                         const ResourceAccessRange &range) const {
    if (!SimpleBinding(buffer)) return HazardResult();
    const auto base_address = ResourceBaseAddress(buffer);
    HazardDetector detector(usage_index);
    return DetectHazardRange(detector, (range + base_address), DetectOptions::kDetectAll);
}

template <typename Detector>
HazardResult AccessContext::DetectHazard(Detector &detector, const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                                         DetectOptions options) const {
    const std::optional<ImageRangeGen> &attachment_gen = view_gen.GetRangeGen(gen_type);
    if (!attachment_gen) return HazardResult();

    ImageRangeGen range_gen(*attachment_gen);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}

template <typename Detector>
HazardResult AccessContext::DetectHazard(Detector &detector, const ImageState &image,
                                         const VkImageSubresourceRange &subresource_range, const VkOffset3D &offset,
                                         const VkExtent3D &extent, bool is_depth_sliced, DetectOptions options) const {
    // range_gen is non-temporary to avoid additional copy
    ImageRangeGen range_gen = image.MakeImageRangeGen(subresource_range, offset, extent, is_depth_sliced);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}

template <typename Detector>
HazardResult AccessContext::DetectHazard(Detector &detector, const ImageState &image,
                                         const VkImageSubresourceRange &subresource_range, bool is_depth_sliced,
                                         DetectOptions options) const {
    // range_gen is non-temporary to avoid additional copy
    ImageRangeGen range_gen = image.MakeImageRangeGen(subresource_range, is_depth_sliced);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}

HazardResult AccessContext::DetectHazard(const ImageState &image, SyncStageAccessIndex current_usage,
                                         const VkImageSubresourceRange &subresource_range, bool is_depth_sliced) const {
    HazardDetector detector(current_usage);
    return DetectHazard(detector, image, subresource_range, is_depth_sliced, DetectOptions::kDetectAll);
}

HazardResult AccessContext::DetectHazard(const ImageViewState &image_view, SyncStageAccessIndex current_usage) const {
    // Get is const, but callee will copy
    HazardDetector detector(current_usage);
    return DetectHazardGeneratedRanges(detector, image_view.GetFullViewImageRangeGen(), DetectOptions::kDetectAll);
}

HazardResult AccessContext::DetectHazard(const ImageRangeGen &ref_range_gen, SyncStageAccessIndex current_usage,
                                         const SyncOrdering ordering_rule) const {
    if (ordering_rule == SyncOrdering::kOrderingNone) {
        HazardDetector detector(current_usage);
        return DetectHazardGeneratedRanges(detector, ref_range_gen, DetectOptions::kDetectAll);
    }

    HazardDetectorWithOrdering detector(current_usage, ordering_rule);
    return DetectHazardGeneratedRanges(detector, ref_range_gen, DetectOptions::kDetectAll);
}

HazardResult AccessContext::DetectHazard(const ImageViewState &image_view, const VkOffset3D &offset, const VkExtent3D &extent,
                                         SyncStageAccessIndex current_usage, SyncOrdering ordering_rule) const {
    // range_gen is non-temporary to avoid an additional copy
    ImageRangeGen range_gen(image_view.MakeImageRangeGen(offset, extent));
    HazardDetectorWithOrdering detector(current_usage, ordering_rule);
    return DetectHazardGeneratedRanges(detector, range_gen, DetectOptions::kDetectAll);
}

HazardResult AccessContext::DetectHazard(const AttachmentViewGen &view_gen, AttachmentViewGen::Gen gen_type,
                                         SyncStageAccessIndex current_usage, SyncOrdering ordering_rule) const {
    HazardDetectorWithOrdering detector(current_usage, ordering_rule);
    return DetectHazard(detector, view_gen, gen_type, DetectOptions::kDetectAll);
}

HazardResult AccessContext::DetectHazard(const ImageState &image, const VkImageSubresourceRange &subresource_range,
                                         const VkOffset3D &offset, const VkExtent3D &extent, bool is_depth_sliced,
                                         SyncStageAccessIndex current_usage, SyncOrdering ordering_rule) const {
    if (ordering_rule == SyncOrdering::kOrderingNone) {
        HazardDetector detector(current_usage);
        return DetectHazard(detector, image, subresource_range, offset, extent, is_depth_sliced, DetectOptions::kDetectAll);
    }
    HazardDetectorWithOrdering detector(current_usage, ordering_rule);
    return DetectHazard(detector, image, subresource_range, offset, extent, is_depth_sliced, DetectOptions::kDetectAll);
}

class BarrierHazardDetector {
  public:
    BarrierHazardDetector(SyncStageAccessIndex usage_index, VkPipelineStageFlags2KHR src_exec_scope,
                          SyncStageAccessFlags src_access_scope)
        : usage_info_(SyncStageAccess::UsageInfo(usage_index)),
          src_exec_scope_(src_exec_scope),
          src_access_scope_(src_access_scope) {}

    HazardResult Detect(const AccessStateConstPtr &state_ptr) const {
        return state_ptr->GetState().DetectBarrierHazard(usage_info_, kQueueIdInvalid, src_exec_scope_, src_access_scope_);
    }
    HazardResult DetectAsync(const AccessStateConstPtr &state_ptr, ResourceUsageTag start_tag) const {
        // Async barrier hazard detection can use the same path as the usage index is not IsRead, but is IsWrite
        return state_ptr->GetState().DetectAsyncHazard(usage_info_, start_tag);
    }

  private:
    const SyncStageAccessInfoType &usage_info_;
    VkPipelineStageFlags2KHR src_exec_scope_;
    SyncStageAccessFlags src_access_scope_;
};

// Use the Detect path for accesses in scope, use Async path for accesses outside of scope
// this isn't quite right, as we need to track the sync tag of out-of-queue accesses, so there's
// some false negatives with this approach
class EventScopeBarrierHazardDetector {
  public:
    EventScopeBarrierHazardDetector(SyncStageAccessIndex usage_index, VkPipelineStageFlags2KHR src_exec_scope,
                                    SyncStageAccessFlags src_access_scope, QueueId queue_id, ResourceUsageTag scope_tag)
        : usage_info_(SyncStageAccess::UsageInfo(usage_index)),
          src_exec_scope_(src_exec_scope),
          src_access_scope_(src_access_scope),
          queue_id_(queue_id) {}

    HazardResult Detect(const AccessStateConstPtr &state_ptr) {
        return state_ptr->GetState().DetectBarrierHazard(usage_info_, queue_id_, src_exec_scope_, src_access_scope_);
    }
    HazardResult DetectAsync(const AccessStateConstPtr &state_ptr, ResourceUsageTag start_tag) const {
        // Async barrier hazard detection can use the same path as the usage index is not IsRead, but is IsWrite
        return state_ptr->GetState().DetectAsyncHazard(usage_info_, start_tag);
    }

  private:
    const SyncStageAccessInfoType usage_info_;
    VkPipelineStageFlags2KHR src_exec_scope_;
    SyncStageAccessFlags src_access_scope_;
    QueueId queue_id_;
};

HazardResult AccessContext::DetectImageBarrierHazard(const ImageState &image, const VkImageSubresourceRange &subresource_range,
                                                     VkPipelineStageFlags2KHR src_exec_scope,
                                                     const SyncStageAccessFlags &src_access_scope, QueueId queue_id,
                                                     const AccessContext &first_scope, const ResourceUsageTag scope_tag,
                                                     AccessContext::DetectOptions options) const {
    ImageRangeGen image_range_gen = image.MakeImageRangeGen(subresource_range, false);

    sparse_container::intersection_generator in_scope_gen(image_range_gen, MapRangeGen(first_scope.access_state_map_));
    EventScopeBarrierHazardDetector detector(SyncStageAccessIndex::SYNC_IMAGE_LAYOUT_TRANSITION, src_exec_scope, src_access_scope,
                                             queue_id, scope_tag);
    HazardResult hazard = first_scope.DetectHazardGeneratedRanges(detector, in_scope_gen, options);
    if (!hazard.IsHazard()) {
        sparse_container::intersection_generator out_scope_gen(image_range_gen, MapRangeGen(access_state_map_));
        hazard = DetectAsyncHazard(detector, out_scope_gen, scope_tag);
    }
    return hazard;
}

HazardResult AccessContext::DetectImageBarrierHazard(const AttachmentViewGen &view_gen, const SyncBarrier &barrier,
                                                     DetectOptions options) const {
    BarrierHazardDetector detector(SyncStageAccessIndex::SYNC_IMAGE_LAYOUT_TRANSITION, barrier.src_exec_scope.exec_scope,
                                   barrier.src_access_scope);
    return DetectHazard(detector, view_gen, AttachmentViewGen::Gen::kViewSubresource, options);
}

HazardResult AccessContext::DetectImageBarrierHazard(const ImageState &image, VkPipelineStageFlags2KHR src_exec_scope,
                                                     const SyncStageAccessFlags &src_access_scope,
                                                     const VkImageSubresourceRange &subresource_range,
                                                     const DetectOptions options) const {
    BarrierHazardDetector detector(SyncStageAccessIndex::SYNC_IMAGE_LAYOUT_TRANSITION, src_exec_scope, src_access_scope);
    return DetectHazard(detector, image, subresource_range, false, options);
}

HazardResult AccessContext::DetectImageBarrierHazard(const SyncImageMemoryBarrier &image_barrier) const {
    return DetectImageBarrierHazard(*image_barrier.image.get(), image_barrier.barrier.src_exec_scope.exec_scope,
                                    image_barrier.barrier.src_access_scope, image_barrier.range, kDetectAll);
}

// This is called with the *recorded* command buffers access context, with the *active* access context pass in, againsts which
// hazards will be detected
HazardResult AccessContext::DetectFirstUseHazard(QueueId queue_id, const ResourceUsageRange &tag_range,
                                                 const AccessContext &access_context) const {
    HazardResult hazard;
    for (const auto &recorded_access : access_state_map_) {
        // Cull any entries not in the current tag range
        if (!recorded_access.second->GetFirstAccess().AccessInTagRange(tag_range)) continue;
        HazardDetectFirstUse detector(recorded_access.second, queue_id, tag_range);
        hazard = access_context.DetectHazardRange(detector, recorded_access.first, DetectOptions::kDetectAll);
        if (hazard.IsHazard()) break;
    }

    return hazard;
}

// For RenderPass time validation this is "start tag", for QueueSubmit, this is the earliest
// unsynchronized tag for the Queue being tested against (max synchrononous + 1, perhaps)
ResourceUsageTag AccessContext::AsyncReference::StartTag() const { return (tag_ == kInvalidTag) ? context_->StartTag() : tag_; }

AttachmentViewGen::AttachmentViewGen(const syncval_state::ImageViewState *image_view, const VkOffset3D &offset,
                                     const VkExtent3D &extent)
    : view_(image_view), view_mask_(image_view->normalized_subresource_range.aspectMask), gen_store_() {
    gen_store_[Gen::kViewSubresource].emplace(image_view->GetFullViewImageRangeGen());
    gen_store_[Gen::kRenderArea].emplace(image_view->MakeImageRangeGen(offset, extent));

    const auto depth = view_mask_ & VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depth && (depth != view_mask_)) {
        gen_store_[Gen::kDepthOnlyRenderArea].emplace(image_view->MakeImageRangeGen(offset, extent, depth));
    }
    const auto stencil = view_mask_ & VK_IMAGE_ASPECT_STENCIL_BIT;
    if (stencil && (stencil != view_mask_)) {
        gen_store_[Gen::kStencilOnlyRenderArea].emplace(image_view->MakeImageRangeGen(offset, extent, stencil));
    }
}

const std::optional<ImageRangeGen> &AttachmentViewGen::GetRangeGen(AttachmentViewGen::Gen type) const {
    static_assert(Gen::kGenSize == 4, "Function written with this assumption");
    // If the view is a depth only view, then the depth only portion of the render area is simply the render area.
    // If the view is a depth stencil view, then the depth only portion of the render area will be a subset,
    // and thus needs the generator function that will produce the address ranges of that subset
    const bool depth_only = (type == kDepthOnlyRenderArea) && (view_mask_ == VK_IMAGE_ASPECT_DEPTH_BIT);
    const bool stencil_only = (type == kStencilOnlyRenderArea) && (view_mask_ == VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth_only || stencil_only) {
        type = Gen::kRenderArea;
    }
    return gen_store_[type];
}

AttachmentViewGen::Gen AttachmentViewGen::GetDepthStencilRenderAreaGenType(bool depth_op, bool stencil_op) const {
    assert(IsValid());
    assert(view_mask_ & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
    if (depth_op) {
        assert(view_mask_ & VK_IMAGE_ASPECT_DEPTH_BIT);
        if (stencil_op) {
            assert(view_mask_ & VK_IMAGE_ASPECT_STENCIL_BIT);
            return kRenderArea;
        }
        return kDepthOnlyRenderArea;
    }
    if (stencil_op) {
        assert(view_mask_ & VK_IMAGE_ASPECT_STENCIL_BIT);
        return kStencilOnlyRenderArea;
    }

    assert(depth_op || stencil_op);
    return kRenderArea;
}

void AccessContext::ResolveAccessMaps(const AccessState::OpContext &op, const ResourceAccessRangeMap &from,
                                      ResourceAccessRangeMap &to) {
    ResourceRangeMergeIterator current(to, from, 0);
    while (current->range.non_empty()) {
        const auto current_range = current->range;
        if (current->pos_B->valid) {
            const auto &src_pos = current->pos_B->lower_bound;
            if (current->pos_A->valid) {
                const auto &dst_pos = current->pos_A->lower_bound;
                // If dst is the resolved state, resolved gets nullptr, s.t. we can skip the move.
                AccessStateConstPtr resolved = AccessState::SSAResolve(op, dst_pos->second, src_pos->second);
                const bool new_state = resolved != dst_pos->second;

                if (dst_pos->first == current_range) {
                    // The current destination covers the current range, so we can just update it
                    // See if we can avoid the update, if the resolved state is the same as the current one
                    if (new_state) {
                        dst_pos->second = std::move(resolved);
                    }
                } else {
                    // The current destination only partially matches current range, so split to only modify
                    // the intersection.

                    // Note: I believe we still need to split because just because the current dst is correct for the current
                    //       piece of src, doesn't mean the next piece will be, and we've been spliting as we go.
                    const auto trimmed = sparse_container::split(current->pos_A->lower_bound, to, current_range);
                    if (new_state) {
                        trimmed->second = std::move(resolved);
                    }
                    current.invalidate_A(trimmed);
                }
            } else {
                // use the from state to infill
                auto inserted = to.insert(current->pos_A->lower_bound, std::make_pair(current_range, src_pos->second));
                current.invalidate_A(inserted);  // Update the parallel iterator to point at the insert segment
            }
        }
        // neither the insert, nor the trim should cause pointing to an empty range
        assert(current->range.non_empty());
        ++current;
    }
}

// A non recursive range walker for the asynchronous contexts (those we have no barriers with)
template <typename Detector, typename RangeGen>
HazardResult AccessContext::DetectAsyncHazard(const Detector &detector, const RangeGen &const_range_gen,
                                              ResourceUsageTag async_tag) const {
    HazardResult hazard;

    sparse_container::intersection_generator range_gen(const_range_gen, MapRangeGen(access_state_map_),
                                                       IntersectionAdvanceSecond());
    for (; range_gen->non_empty(); ++range_gen) {
        hazard = detector.DetectAsync(range_gen.get_second_gen().get_pos()->second, async_tag);
        if (hazard.IsHazard()) break;
    }

    return hazard;
}

// A recursive range walker for hazard detection, first for the current context and the (DetectHazardRecur) to walk
// the DAG of the contexts (for example subpasses)
template <typename Detector, typename RangeGen>
HazardResult AccessContext::DetectHazardGeneratedRanges(Detector &detector, RangeGen &range_gen, DetectOptions options) const {
    HazardResult hazard;

    // Do this before range_gen is incremented s.t. the copies used will be correct
    if (static_cast<uint32_t>(options) & DetectOptions::kDetectAsync) {
        // Async checks don't require recursive lookups, as the async lists are exhaustive for the top-level context
        // so we'll check these first
        for (const auto &async_ref : async_) {
            hazard = async_ref.Context().DetectAsyncHazard(detector, range_gen, async_ref.StartTag());
            if (hazard.IsHazard()) return hazard;
        }
    }

    sparse_container::mapped_range_generator map_gen(access_state_map_);
    sparse_container::intersection_generator hazard_gen(range_gen, map_gen);
    for (; !hazard_gen->empty(); ++hazard_gen) {
        hazard = detector.Detect(hazard_gen.get_second_gen().get_pos()->second);
        if (hazard.IsHazard()) return hazard;
    }

    const bool detect_prev = (static_cast<uint32_t>(options) & DetectOptions::kDetectPrevious) != 0;
    if (detect_prev && (prev_.size() > 0)) {
        // PrevAccess only looks in locations where there are no values (i.e. in gaps in the current map)
        auto prev_access_map = ResolvePreviousAccessRangeGen(range_gen, MakeOpContext(), nullptr);
        for (const auto &prev : prev_access_map) {
            hazard = detector.Detect(prev.second);
            if (hazard.IsHazard()) return hazard;
        }
    }

    return hazard;
}
ResourceAccessRangeMap AccessContext::ResolvePreviousAccessRangeGen(const RangeGenWrapper &gen_arg,
                                                                    const AccessState::OpContext &op,
                                                                    const Ops::ApplyBarrierStack *previous_barrier) const {
    if (prev_.size() == 0) {
        assert(prev_.size() > 0);
    }

    // Look for something to fill the gap further along.
    const bool single_previous = prev_.size() == 1;

    // Use an intersection generator to only recursively search in the gaps to the current map
    sparse_container::intersection_generator prev_gen(sparse_container::mapped_gap_generator(access_state_map_), gen_arg);

    auto recur_prev = [this, &gen_arg, previous_barrier, &op, &prev_gen](const TrackBack &prev_dep) {
        const Ops::ApplyBarrierStack barrier_action(prev_dep.barriers, previous_barrier);
        AccessState::OpContext recur_op(op);
        return prev_dep.source_subpass->ResolveAccessRangeGen(prev_gen, barrier_action, recur_op, true);
    };

    // To get RVO -- need to return an auto scope variable -- so we treat the first one specially
    ResourceAccessRangeMap resolved(recur_prev(prev_[0]));

    if (single_previous) {
        // Nothing to reduce with... thus, done.
        return resolved;
    }

    // Otherwise we need to build a reducible array of prev to resolve.
    std::vector<ResourceAccessRangeMap *> resolve_maps;
    resolve_maps.reserve(prev_.size());

    std::vector<ResourceAccessRangeMap> previous;
    previous.reserve(prev_.size() - 1);

    // This the general case
    for (const auto &prev_dep : prev_) {
        if (resolve_maps.empty()) {
            // Treating the 0'th entry as special
            resolve_maps.push_back(&resolved);
        } else {
            // Since previous is has a capacity of prev_.size() -1, references stored in resolve_maps are not invalidated.
            assert(previous.size() < previous.capacity());
            previous.emplace_back(recur_prev(prev_dep));
            resolve_maps.push_back(&previous.back());
        }
    }

    AccessState::OpContext resolve_op(op);
    while (resolve_maps.size() > 1) {
        size_t size = resolve_maps.size();
        size_t limit = size - 1;
        size_t half = size / 2;
        for (size_t lower = 0; lower < half; lower++) {
            // Result is second map
            ResolveAccessMaps(resolve_op, *resolve_maps[limit - lower], *resolve_maps[lower]);
        }
        // Get rid of the half that was resolved (
        resolve_maps.resize(size - half);
    }
    return resolved;
}

template <typename ResolveOp>
void AccessContext::ResolveFromContextImpl(ResolveOp &&resolve_op, const AccessContext &from_context, bool infill_default,
                                           bool recur_to_infill) {
    ResolveFromContextImpl(resolve_op, SingleRangeGen(kFullRange), from_context, infill_default, recur_to_infill);
}

template <typename ResolveOp, typename RangeGenerator>
void AccessContext::ResolveFromContextImpl(const ResolveOp &resolve_op, const RangeGenerator &range_gen,
                                           const AccessContext &from_context, bool infill_default, bool recur_to_infill) {
    ResourceAccessRangeMap resolve_from =
        from_context.ResolveAccessRangeGen(range_gen, resolve_op, MakeOpContext(), recur_to_infill);
    AccessState::OpContext op_context = MakeOpContext();
    ResolveAccessMaps(op_context, resolve_from, access_state_map_);
    if (infill_default) {
        // TODO Memoize infill state
        AccessStateConstPtr infill_state(resolve_op.Update(op_context, AccessState::DefaultAccessState()));
        InfillGapRangeGen(range_gen, infill_state);
    }
}

template <typename RangeGen, typename BarrierAction>
ResourceAccessRangeMap AccessContext::ResolveAccessRangeGenNoRecur(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                                   const AccessState::OpContext &op) const {
    ResourceAccessRangeMap resolve_map;
    sparse_container::intersection_generator range_gen(gen_arg, MapRangeGen(access_state_map_));
    for (; !range_gen->empty(); ++range_gen) {
        const auto &pos = range_gen.get_second_gen().get_pos();
        // Copy and apply barrier
        AccessStateConstPtr insert_value(pos->second);
        AccessState::SSAUpdate(op, barrier_action, insert_value);
        resolve_map.insert(resolve_map.end(), std::make_pair(*range_gen, std::move(insert_value)));
    }
    return resolve_map;
}

template <typename RangeGen, typename BarrierAction>
ResourceAccessRangeMap AccessContext::ResolveAccessRangeGen(const RangeGen &gen_arg, BarrierAction &barrier_action,
                                                            const AccessState::OpContext &op, bool recur_to_infill) const {
    ResourceAccessRangeMap resolve_map(ResolveAccessRangeGenNoRecur(gen_arg, barrier_action, op));

    // Infill if range goes passed both the current and resolve map prior contents
    if (recur_to_infill && (prev_.size() > 0)) {
        // This generator is only contains the ranges in gen_arg that are not present in the current map
        RangeGenWrapper recur_gen(sparse_container::intersection_generator(MapGapGen(resolve_map), gen_arg));

        // ResolvePrevious constructs it's own OpContext's so no need to do so before calling
        ResourceAccessRangeMap prev_map = ResolvePreviousAccessRangeGen(recur_gen, op);
        // Apply the non-stacked barrier action
        ForAllRangeMap(barrier_action, AccessState::OpContext(op), prev_map);

        if (!prev_map.empty()) {
            // Resolves to last argument, and needs it's own op context
            ResolveAccessMaps(AccessState::OpContext(op), prev_map, resolve_map);
        }
    }
    return resolve_map;
}

template <typename RangeGen>
ResourceAccessRangeMap AccessContext::ResolveAccessRangeGen(const RangeGen &gen_arg, Ops::ApplyBarrierStack &barrier_action,
                                                            const AccessState::OpContext &op, bool recur_to_infill) const {
    ResourceAccessRangeMap resolve_map(ResolveAccessRangeGenNoRecur(gen_arg, barrier_action, op));

    // Infill if range goes passed both the current and resolve map prior contents
    if (recur_to_infill && (prev_.size() > 0)) {
        // This generator is only contains the ranges in gen_arg that are not present in the current map
        RangeGenWrapper recur_gen(sparse_container::intersection_generator(MapGapGen(resolve_map), gen_arg));

        // ResolvePrevious constructs it's own OpContext's so no need to do so before calling
        const ResourceAccessRangeMap prev_map = ResolvePreviousAccessRangeGen(recur_gen, op, &barrier_action);

        if (!prev_map.empty()) {
            // Resolves to last argument, and needs it's own op context
            ResolveAccessMaps(AccessState::OpContext(op), prev_map, resolve_map);
        }
    }
    return resolve_map;
}

template <typename Predicate>
struct DoApplyPredicatedWait {
    const Predicate &predicate;
    bool operator()(ResourceAccessState &state) const { return state.ApplyPredicatedWait<Predicate>(predicate); };
};

template <typename Predicate>
void AccessContext::ApplyPredicatedWait(const Predicate &predicate) {
    AccessState::OpContext op_context = MakeOpContext();
    DoApplyPredicatedWait<Predicate> transform{predicate};
    // The CullingUpdater sets entries returning true from transform to OpContext::DefaultState
    Ops::ApplyCullingUpdateState<DoApplyPredicatedWait<Predicate>> updater{transform};
    ForAll(updater);

    // Now purge all default value states from the map.
    const AccessStateConstPtr &default_state = AccessState::DefaultAccessState();
    vvl::EraseIf(access_state_map_,
                 [&default_state](const ResourceAccessRangeMap::value_type &entry) { return entry.second == default_state; });
}

void AccessContext::ApplyAcquireWait(ResourceUsageTag present_tag, ResourceUsageTag acquire_tag) {
    ResourceAccessState::WaitAcquirePredicate predicate{present_tag, acquire_tag};
    ApplyPredicatedWait(predicate);
}

void AccessContext::ApplyTaggedWait(QueueId queue_id, ResourceUsageTag tag) {
    const bool any_queue = (queue_id == kQueueAny);

    if (any_queue) {
        // This isn't just avoid an unneeded test, but to allow *all* queues to to be waited in a single pass
        // (and it does avoid doing the same test for every access, as well as avoiding the need for the predicate
        // to grok Queue/Device/Wait differences.
        ResourceAccessState::WaitTagPredicate predicate{tag};
        ApplyPredicatedWait(predicate);
    } else {
        ResourceAccessState::WaitQueueTagPredicate predicate{queue_id, tag};
        ApplyPredicatedWait(predicate);
    }
}

AccessStateConstPtr AccessState::SSAUpdateAccess(const OpContext &op_context, const AccessStateConstPtr &from,
                                                 const SyncStageAccessInfoType &usage_info, const SyncOrdering ordering_rule,
                                                 const ResourceUsageTag tag) {
    // TODO Memoize state_ update
    auto update_state = std::make_shared<ResourceAccessState>(*from->state_);

    // WIP TODO Use "write overwrites" logic here
    // WIP TODO Make ResourceAccessState tags relative to an AccessState tag
    update_state->Update(usage_info, ordering_rule, tag);

    // Update first iff not closed
    if (!from->first_->closed) {
        FirstPtr update_first = std::make_shared<FirstAccesses>(*from->first_);
        update_first->Update(tag, usage_info, update_state->GetExecutionBarriers(), ordering_rule);
        return std::make_shared<const AccessState>(std::move(update_state), std::move(update_first));
    }
    return std::make_shared<const AccessState>(*from, std::move(update_state));
}

template <typename Updater>
AccessStateConstPtr AccessState::SSAUpdateStateAndFirst(const OpContext &op_context, const AccessStateConstPtr &from,
                                                        const Updater &updater) {
    // TODO Memoize update
    auto update_state = std::make_shared<ResourceAccessState>(*from->state_);
    updater.Update(*update_state);

    auto update_first = std::make_shared<FirstAccesses>(*from->first_);
    updater.Update(*update_first);

    return std::make_shared<const AccessState>(std::move(update_state), std::move(update_first));
}

template <typename Scope, typename BarrierIt>
AccessStateConstPtr AccessState::SSAApplyScopedBarriers(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                        const BarrierIt begin, const BarrierIt end, bool layout_transition,
                                                        ResourceUsageTag tag) {
    std::optional<PendingBarriers> immediate_pending;
    if (from.pending_) {
        // NOTE: Possible additional memoization here:
        immediate_pending.emplace(*from.pending_);
    } else {
        // Reserve the needed
        immediate_pending.emplace(*from.state_);
    }

    for (BarrierIt barrier = begin; barrier != end; ++barrier) {
        from.state_->ApplyBarrier(scope, *barrier, layout_transition, *immediate_pending);
    }

    return SSAApplyPendingBarriers(op_context, from, tag, *immediate_pending);
}

template <typename Scope>
AccessStateConstPtr AccessState::SSAApplyBarrierStack(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                      const SyncBarrierStack &barrier_stack) {
    assert(!from.pending_);

    // TODO Memoize state_ update
    StatePtr update_state = std::make_shared<ResourceAccessState>(*from.state_);
    ResourceAccessState &state = *update_state;
    // If we don't see an layout transition, we won't need a new first.
    FirstPtr update_first;
    const bool first_closed = from.first_->closed;

    const SyncBarrierStack *current = &barrier_stack;
    while (current) {
        PendingBarriers pending(*from.state_);
        for (const auto &barrier : current->barriers) {
            state.ApplyBarrier(scope, barrier, current->layout_transition, pending);
        }

        state.ApplyPendingBarriers(current->tag, pending);

        // First only cares about the layout transition write op
        if (current->layout_transition && !first_closed) {
            const SyncStageAccessInfoType &layout_usage_info = SyncStageAccess::UsageInfo(SYNC_IMAGE_LAYOUT_TRANSITION);
            if (!update_first) {
                update_first = std::make_shared<FirstAccesses>(*from.first_);
            }
            update_first->Update(current->tag, layout_usage_info, state.GetExecutionBarriers(), SyncOrdering::kNonAttachment);
            update_first->TouchupForLayoutTransition(current->tag, pending.GetLayoutOrdering());
        }
        current = current->prev;
    }

    if (update_first) {
        return std::make_shared<const AccessState>(std::move(update_state), std::move(update_first));
    }
    return std::make_shared<const AccessState>(from, std::move(update_state));
}

template <typename Scope, typename BarrierIt>
AccessStateConstPtr AccessState::SSAUpdateScopedBarriers(const Scope &scope, const OpContext &op_context, const AccessState &from,
                                                         const BarrierIt begin, const BarrierIt end, bool layout_transition) {
    // Inefficient to call if begin == end (or barriers are noops), but caller should short circuit

    PendingPtr update_pending;
    if (from.pending_) {
        update_pending = std::make_shared<PendingBarriers>(*from.pending_);
    } else {
        update_pending = std::make_shared<PendingBarriers>(*from.state_);
    }

    assert(update_pending);
    PendingBarriers &pending = *update_pending;

    const ResourceAccessState &state = *from.state_;
    for (BarrierIt barrier = begin; barrier != end; ++barrier) {
        state.ApplyBarrier(scope, *barrier, layout_transition, pending);
    }
    return std::make_shared<const AccessState>(from, std::move(update_pending));
}

AccessStateConstPtr AccessState::SSAApplyPendingBarriers(const OpContext &op_context, const AccessState &from,
                                                         ResourceUsageTag tag) {
    if (from.pending_) {
        return SSAApplyPendingBarriers(op_context, from, tag, *from.pending_);
    }
    return AccessStateConstPtr();
}

AccessStateConstPtr AccessState::SSAApplyPendingBarriers(const OpContext &op_context, const AccessState &from, ResourceUsageTag tag,
                                                         const PendingBarriers &pending) {
    // TODO Memoize state_ update
    StatePtr update_state = std::make_shared<ResourceAccessState>(*from.state_);
    update_state->ApplyPendingBarriers(tag, pending);

    // First only cares about the layout transition write op
    if (pending.HasLayoutTransition() && !from.first_->closed) {
        const SyncStageAccessInfoType &layout_usage_info = SyncStageAccess::UsageInfo(SYNC_IMAGE_LAYOUT_TRANSITION);
        FirstPtr update_first = std::make_shared<FirstAccesses>(*from.first_);
        update_first->Update(tag, layout_usage_info, update_state->GetExecutionBarriers(), SyncOrdering::kNonAttachment);
        update_first->TouchupForLayoutTransition(tag, pending.GetLayoutOrdering());
        return std::make_shared<const AccessState>(std::move(update_state), std::move(update_first));
    }

    return std::make_shared<const AccessState>(from, std::move(update_state));
}

AccessStateConstPtr AccessState::SSAApplySemaphoreBarriers(const OpContext &op_context, const AccessState &from,
                                                           const SemaphoreScope &signal, const SemaphoreScope wait) {
    // TODO Memoize state_ update
    StatePtr update_state = std::make_shared<ResourceAccessState>(*from.state_);
    update_state->ApplySemaphore(signal, wait);
    return std::make_shared<const AccessState>(from, std::move(update_state));
}

AccessStateConstPtr AccessState::SSANormalizeAndClearFirstUse(const OpContext &op_context, const AccessState &from) {
    if (from.GetState().LastReadsSize() > 1 || from.first_) {
        StatePtr update_state = std::make_shared<ResourceAccessState>(*from.state_);
        update_state->Normalize();
        return std::make_shared<const AccessState>(std::move(update_state), FirstAccesses::ForceClosed());
    }
    return AccessStateConstPtr();
}
template <typename CullingTransform>
AccessStateConstPtr AccessState::SSACullingUpdateState(const OpContext &op_context, const AccessState &from,
                                                       const CullingTransform &transform) {
    StatePtr update_state = std::make_shared<ResourceAccessState>(*from.state_);
    bool cull = transform(*update_state);
    if (cull) {
        return DefaultAccessState();
    }

    return std::make_shared<const AccessState>(from, std::move(update_state));
}

#if 0
const AccessStateConstPtr &AccessState::OpContext::DefaultState() const {
    return context_.DefaultAccessState();
    // TODO: insert return statement here
}
#endif

AccessState::OpID AccessState::OpContext::Op() const {
    return op_id_;
    // TODO: insert return statement here
}

AccessState::OpContext::OpContext(const AccessContext &access_context) : context_(access_context), op_id_(++next_op_) {}

AccessState::OpContext::OpContext(const OpContext &other_context) : OpContext(other_context.context_) {}

AccessStateConstPtr AccessContext::Ops::OffsetTags::Update(const AccessState::OpContext &op_context,
                                                           const AccessStateConstPtr &from) const {
    return AccessState::SSAUpdateStateAndFirst(op_context, from, *this);
}

AccessContext::Ops::ApplyAcquireNextSemaphore::ApplyAcquireNextSemaphore(const SyncExecScope &wait_scope,
                                                                         ResourceUsageTag acquire_tag_)
    : barrier(getPresentSrcScope(), getPresentValidAccesses(), wait_scope, SyncStageAccessFlags()), acquire_tag(acquire_tag_) {}

AccessStateConstPtr AccessContext::Ops::ApplyAcquireNextSemaphore::Update(const AccessState::OpContext &op_context,
                                                                          const AccessStateConstPtr &from) const {
    if (from->GetState().LastWriteTag() <= acquire_tag) {
        return AccessState::SSAApplyScopedBarriers(ResourceAccessState::UntaggedScopeOps(), op_context, *from, &barrier,
                                                   &barrier + 1, false, kInvalidTag);
    }
    return AccessStateConstPtr();
}

// kPresentSrcScope/kPresentValidAccesses cannot be regular global variables, because they use global
// variables from another compilation unit (through syncStageAccessMaskByStageBit() call) for initialization,
// and initialization of globals between compilation units is undefined. Instead they get initialized
// on the first use (it's important to ensure this first use is also not initialization of some global!).

const SyncExecScope &AccessContext::Ops::ApplyAcquireNextSemaphore::getPresentSrcScope() const {
    static const SyncExecScope kPresentSrcScope =
        SyncExecScope(VK_PIPELINE_STAGE_2_PRESENT_ENGINE_BIT_SYNCVAL,  // mask_param (unused)
                      VK_PIPELINE_STAGE_2_PRESENT_ENGINE_BIT_SYNCVAL,  // expanded_mask
                      VK_PIPELINE_STAGE_2_PRESENT_ENGINE_BIT_SYNCVAL,  // exec_scope
                      getPresentValidAccesses());                      // valid_accesses
    return kPresentSrcScope;
}

const SyncStageAccessFlags &AccessContext::Ops::ApplyAcquireNextSemaphore::getPresentValidAccesses() const {
    static const SyncStageAccessFlags kPresentValidAccesses =
        SyncStageAccessFlags(SyncStageAccess::AccessScopeByStage(VK_PIPELINE_STAGE_2_PRESENT_ENGINE_BIT_SYNCVAL));
    return kPresentValidAccesses;
}

AccessStateConstPtr AccessContext::Ops::UpdateAccess::Update(const AccessState::OpContext &op_context,
                                                             const AccessStateConstPtr &from) const {
    assert(usage_info.stage_access_index != SYNC_ACCESS_INDEX_NONE);
    return AccessState::SSAUpdateAccess(op_context, from, usage_info, ordering_rule, tag);
}

AccessStateConstPtr AccessContext::Ops::NormalizeAndClearFirstUse::Update(const AccessState::OpContext &op_context,
                                                                          const AccessStateConstPtr &from) const {
    return AccessState::SSANormalizeAndClearFirstUse(op_context, *from);
}

AccessStateConstPtr AccessContext::Ops::ApplySemaphoreBarrier::Update(const AccessState::OpContext &op_context,
                                                                      const AccessStateConstPtr &from) const {
    return AccessState::SSAApplySemaphoreBarriers(op_context, *from, signal, wait);
}

AccessContext::Ops::ApplyRenderPassLayoutTransition::ApplyRenderPassLayoutTransition(const std::vector<SyncBarrier> &barriers_,
                                                                                     ResourceUsageTag tag_)
    : barriers(barriers_), scope(), tag(tag_) {}

AccessStateConstPtr AccessContext::Ops::ApplyRenderPassLayoutTransition::Update(const AccessState::OpContext &op_context,
                                                                                const AccessStateConstPtr &from) const {
    assert(!barriers.empty());
    return AccessState::SSAApplyScopedBarriers(scope, op_context, *from, barriers.begin(), barriers.end(), true, tag);
}

AccessContext::Ops::ApplyBarrierStack::ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const bool layout_transition,
                                                         const ResourceUsageTag tag, const SyncBarrierStack *prev)
    : scope(), barrier_stack{barriers, layout_transition, tag, prev} {}
AccessContext::Ops::ApplyBarrierStack::ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const bool layout_transition,
                                                         const ResourceUsageTag tag, const ApplyBarrierStack &prev)
    : ApplyBarrierStack(barriers, layout_transition, tag, &prev.barrier_stack) {}
AccessContext::Ops::ApplyBarrierStack::ApplyBarrierStack(const std::vector<SyncBarrier> &barriers, const ApplyBarrierStack *prev)
    : ApplyBarrierStack(barriers, false, kInvalidTag, (prev ? &prev->barrier_stack : nullptr)) {}
// ApplyBarrierStack(barriers, false, kInvalidTag, (prev ? &prev->barrier_stack : static_cast<const SyncBarrierStack *>(nullptr)))
// {}
AccessStateConstPtr AccessContext::Ops::ApplyBarrierStack::Update(const AccessState::OpContext &op_context,
                                                                  const AccessStateConstPtr &from) const {
    return AccessState::SSAApplyBarrierStack(scope, op_context, *from, barrier_stack);
}

bool AccessContext::Ops::ApplyQueueScopeBarrier::InitLayoutTransition(QueueId queue_id, bool layout_transition) {
    return (queue_id != kQueueIdInvalid) ? false : layout_transition;
}

AccessContext::Ops::ApplyQueueScopeBarrier::ApplyQueueScopeBarrier(QueueId queue_id, const SyncBarrier &barrier_,
                                                                   bool layout_transition_, ResourceUsageTag tag_)
    : barrier(barrier_), scope(queue_id), tag(tag_), layout_transition(InitLayoutTransition(queue_id, layout_transition_)) {
    assert(!layout_transition || (tag != kInvalidTag));
}

AccessStateConstPtr AccessContext::Ops::ApplyQueueScopeBarrier::Update(const AccessState::OpContext &op_context,
                                                                       const AccessStateConstPtr &from) const {
    return AccessState::SSAApplyScopedBarriers(scope, op_context, *from, &barrier, &barrier + 1, layout_transition, tag);
}

AccessStateConstPtr AccessContext::Ops::ApplyPendingBarriers::Update(const AccessState::OpContext &op_context,
                                                                     const AccessStateConstPtr &from) const {
    return AccessState::SSAApplyPendingBarriers(op_context, *from, tag);
}

AccessStateConstPtr AccessContext::Ops::SetQueueOffsetTags::Update(const AccessState::OpContext &op_context,
                                                                   const AccessStateConstPtr &from) const {
    return AccessState::SSAUpdateStateAndFirst(op_context, from, *this);
}

void AccessContext::Ops::SetQueueOffsetTags::Update(ResourceAccessState &state) const {
    state.OffsetTag(offset);
    state.SetQueueId(queue);
}

void AccessContext::Ops::SetQueueOffsetTags::Update(FirstAccesses &first) const { first.OffsetTag(offset); }

AccessStateConstPtr AccessContext::Ops::UpdatePendingEventBarriers::Update(const AccessState::OpContext &op_context,
                                                                           const AccessStateConstPtr &from) const {
    return AccessState::SSAUpdateScopedBarriers(scope, op_context, *from, &barrier, &barrier + 1, layout_transition);
}

AccessStateConstPtr AccessContext::Ops::UpdatePendingGlobalEventBarriers::Update(const AccessState::OpContext &op_context,
                                                                                 const AccessStateConstPtr &from) const {
    if (!barriers.empty()) {
        return AccessState::SSAUpdateScopedBarriers(scope, op_context, *from, barriers.begin(), barriers.end(), false);
    }
    return AccessStateConstPtr();
}

AccessStateConstPtr AccessContext::Ops::UpdateAndApplyGlobalPipelineBarriers::Update(const AccessState::OpContext &op_context,
                                                                                     const AccessStateConstPtr &from) const {
    if (barriers.empty()) {
        return AccessState::SSAApplyPendingBarriers(op_context, *from, tag);
    }
    return AccessState::SSAApplyScopedBarriers(scope, op_context, *from, barriers.begin(), barriers.end(), false, tag);
}

AccessStateConstPtr AccessContext::Ops::UpdatePendingPipelineBarriers::Update(const AccessState::OpContext &op_context,
                                                                              const AccessStateConstPtr &from) const {
    return AccessState::SSAUpdateScopedBarriers(scope, op_context, *from, &barrier, &barrier + 1, layout_transition);
}

template <typename CullingTransform>
inline AccessStateConstPtr AccessContext::Ops::ApplyCullingUpdateState<CullingTransform>::Update(
    const AccessState::OpContext &op_context, const AccessStateConstPtr &from) const {
    return AccessState::SSACullingUpdateState(op_context, *from, transform);
}

template <typename UpdateAction>
class SSAUpdateOperations {
  public:
    using Map = ResourceAccessRangeMap;
    using Range = typename Map::key_type;
    using Iterator = typename Map::iterator;

    void update(const Iterator &pos) const {
        auto &access_state = pos->second;
        // SSAUpdate treats the 2nd arg as read/write, and updates based on the results of the updater
        AccessState::SSAUpdate(op_context_, updater_, access_state);
    }

    void infill(Map &accesses, const Iterator &pos, const Range &infill_range) const {
        if (infill_state_) {
            accesses.insert(pos, std::make_pair(infill_range, infill_state_));
        }
    }

    SSAUpdateOperations(const UpdateAction &update_action, const AccessState::OpContext &op_context, bool infill_default)
        : updater_(update_action), op_context_(op_context), infill_state_() {
        if (infill_default) {
            // SSAUpdate treats the 2nd arg as read/write, and updates based on the results of the updater
            infill_state_ = AccessState::DefaultAccessState();
            AccessState::SSAUpdate(op_context, updater_, infill_state_);
        }
    }

  private:
    const UpdateAction &updater_;
    const AccessState::OpContext &op_context_;
    AccessStateConstPtr infill_state_;
};

template <typename Action>
void AccessContext::UpdateMemoryAccessRangeState(Action &action, const ResourceAccessRange &range, bool infill_default) {
    UpdateMemoryAccessState(action, sparse_container::single_range_generator(range), infill_default);
}

// Infill default is needed:
//    All updates that affect state_ or first_
//    All all barriers that contain layout transitions (and thus will affect state)
template <typename Action, typename RangeGen>
void AccessContext::UpdateMemoryAccessState(const Action &action, const RangeGen &range_gen, bool infill_default) {
    // Resolve if required
    if (prev_.size() > 0) {
        // Only resolves in the intersection of range_gen and the gaps in access_state_map_
        auto prev_map = ResolvePreviousAccessRangeGen(range_gen, MakeOpContext(), nullptr);
        if (!prev_map.empty()) {
            sparse_container::merge_non_intersecting(access_state_map_, prev_map.begin(), prev_map.end(),
                                                     sparse_container::assign_policy_move());
        }
    }

    SSAUpdateOperations update_operations(action, MakeOpContext(), infill_default);
    RangeGen gen_copy(range_gen);
    sparse_container::infill_update_rangegen(access_state_map_, gen_copy, update_operations);
}

template <typename Detector>
HazardResult AccessContext::DetectHazardRange(Detector &detector, const ResourceAccessRange &range, DetectOptions options) const {
    sparse_container::single_range_generator range_gen(range);
    return DetectHazardGeneratedRanges(detector, range_gen, options);
}
