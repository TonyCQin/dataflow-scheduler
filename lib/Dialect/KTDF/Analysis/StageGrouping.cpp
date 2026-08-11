//===-- StageGrouping.cpp ---------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the stage grouping analysis.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGrouping.h"

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "stage-grouping"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// StageGroupingAnalysis Implementation
//===----------------------------------------------------------------------===//

StageGroupingAnalysis::StageGroupingAnalysis(
    PipelineOp pipeline, const scheduler::arch_view::MemoryTree& memory_tree)
    : memory_tree_(memory_tree) {
  auto stages_range = pipeline.getStages();
  init(llvm::SmallVector<StageOp>(stages_range.begin(), stages_range.end()));
}

StageGroupingAnalysis::StageGroupingAnalysis(
    llvm::ArrayRef<StageOp> stages,
    const scheduler::arch_view::MemoryTree& memory_tree)
    : memory_tree_(memory_tree) {
  init(stages);
}

void StageGroupingAnalysis::init(llvm::ArrayRef<StageOp> stages) {
  if (stages.empty()) return;

  // Use EquivalenceClasses to group stages based on shared resources
  llvm::EquivalenceClasses<unsigned> ec;

  // Initialize: each stage starts in its own equivalence class
  for (size_t i = 0; i < stages.size(); ++i) {
    ec.insert(i);
  }

  // Merge equivalence classes for stages that should be grouped together
  for (size_t i = 0; i < stages.size(); ++i) {
    for (size_t j = i + 1; j < stages.size(); ++j) {
      if (shouldGroupTogether(stages[i], stages[j])) {
        ec.unionSets(i, j);
      }
    }
  }

  // Build groups by iterating over equivalence classes
  // Only process leader entries to avoid creating empty groups
  unsigned group_id = 0;

  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    // Skip non-leader entries
    if (!(*it)->isLeader()) continue;

    // Create a new group for this equivalence class
    StageGroup group;

    // Iterate over all members of this equivalence class
    for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
      unsigned stage_idx = *mi;
      group.addStage(stages[stage_idx]);

      StageOp stage = stages[stage_idx];
      stage_to_group_[stage.getOperation()] = group_id;
    }

    groups_.push_back(group);
    group_id++;
  }
  LDBG(1) << "Stage grouping analysis complete";
  LDBG(1) << "  Total groups: " << groups_.size();
}

const StageGroup* StageGroupingAnalysis::getGroupForStage(StageOp stage) const {
  auto it = stage_to_group_.find(stage.getOperation());
  if (it == stage_to_group_.end()) return nullptr;
  return &groups_[it->second];
}

bool StageGroupingAnalysis::areInSameGroup(StageOp stage1,
                                           StageOp stage2) const {
  auto it1 = stage_to_group_.find(stage1.getOperation());
  auto it2 = stage_to_group_.find(stage2.getOperation());

  if (it1 == stage_to_group_.end() || it2 == stage_to_group_.end())
    return false;

  return it1->second == it2->second;
}

void StageGroupingAnalysis::print(llvm::raw_ostream& os) const {
  os << "  Total groups: " << groups_.size() << "\n\n";

  for (size_t i = 0; i < groups_.size(); ++i) {
    const StageGroup& group = groups_[i];
    os << "  Group " << i << ":\n";
    os << "    Stages: [";
    bool first = true;
    for (StageOp stage : group.getStages()) {
      if (!first) os << ", ";
      os << "stage ";
      scheduler::printLocation(os, stage.getOperation());
      first = false;
    }
    os << "]\n";
    os << "    Single stage: " << (group.isSingleStage() ? "yes" : "no")
       << "\n";
  }
}

void StageGroupingAnalysis::dump() const {
  llvm::dbgs() << "Stage Grouping Analysis Results:\n";
  print(llvm::dbgs());
}

bool StageGroupingAnalysis::shouldGroupTogether(StageOp stage1,
                                                StageOp stage2) {
  // Analyze ALL operations in both stages
  // Group stages that:
  // 1. Access the same memory below L1 in the hierarchy (due to size
  // constraint)
  // 2. Access the same FIFO types (just the enum, not sizes/element types)

  llvm::DenseSet<mlir::Attribute> stage1_below_scratchpad_mems;
  llvm::DenseSet<mlir::Attribute> stage2_below_scratchpad_mems;
  llvm::SmallVector<FifoSlotType, 4> stage1_fifos;
  llvm::SmallVector<FifoSlotType, 4> stage2_fifos;

  // Helper to analyze a value for memories below scratchpad or FIFO types
  auto analyzeValue =
      [&](Value val, llvm::DenseSet<mlir::Attribute>& below_scratchpad_mems,
          llvm::SmallVector<FifoSlotType, 4>& fifos) {
        if (auto memref_type = dyn_cast<MemRefType>(val.getType())) {
          auto mem_space = memref_type.getMemorySpace();
          if (mem_space && memory_tree_.isBelowScratchPad(mem_space)) {
            below_scratchpad_mems.insert(mem_space);
          }
        } else if (auto fifo_slot_type =
                       dyn_cast<FifoSlotType>(val.getType())) {
          fifos.push_back(fifo_slot_type);
        }
      };

  // Analyze ALL operations in stage1 - check all operands
  stage1.walk([&](Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage1_below_scratchpad_mems, stage1_fifos);
    }
  });

  // Analyze ALL operations in stage2 - check all operands
  stage2.walk([&](Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage2_below_scratchpad_mems, stage2_fifos);
    }
  });

  // Group if they share any memory below scratchpad
  for (mlir::Attribute mem : stage1_below_scratchpad_mems) {
    if (stage2_below_scratchpad_mems.contains(mem)) {
      return true;
    }
  }

  // Group if they share any FIFO types (comparing src and dest attributes)
  for (FifoSlotType fifo1 : stage1_fifos) {
    for (FifoSlotType fifo2 : stage2_fifos) {
      // Two FIFOs are considered the same if they have the same src and dest
      if (fifo1.getSrc() == fifo2.getSrc() &&
          fifo1.getDest() == fifo2.getDest()) {
        return true;
      }
    }
  }

  return false;
}

// Made with Bob
