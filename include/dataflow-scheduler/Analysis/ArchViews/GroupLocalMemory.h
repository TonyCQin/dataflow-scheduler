//===-- GroupLocalMemory.h --------------------------------------*- c++ -*-===//
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
// GroupLocalMemory
//
// Captures the local memory resource present directly inside of a
// ktdf_arch.group across the different groups.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Attributes.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

namespace scheduler::arch_view {

/// Maps exec_unit kind -> local memory kind for groups that contain both.
///
/// During construction the set of memory kinds for each exec_unit kind is
/// intersected across all groups that contain it — only memory kinds present
/// in every such group are retained.  Ambiguity (intersection size > 1) is
/// not diagnosed at construction time; it becomes a hard error only if that
/// exec_unit kind is queried via getLocalMemoryKindForStage.
///
/// Constructed as a DeviceView child of a DeviceOp.
class GroupLocalMemory : public mlir::ktdf_arch::DeviceView {
 public:
  explicit GroupLocalMemory(const mlir::ktdf_arch::Device& device);

  /// Returns the kind attribute of the local memory co-located with the
  /// exec_unit identified by @p exec_unit_kind, or nullptr if no
  /// unambiguous mapping exists.
  [[nodiscard]] mlir::Attribute getLocalMemoryKind(
      mlir::Attribute exec_unit_kind) const;

  /// Returns the local memory kind for the single exec_unit kind declared on
  /// @p stage.
  ///
  /// Emits an error on @p stage and returns nullptr if:
  ///   - the stage does not have exactly one applicable exec_unit, or
  ///   - no unambiguous local memory is mapped to that exec_unit kind.
  [[nodiscard]] mlir::Attribute getLocalMemoryKindForStage(
      mlir::ktdf::StageOp stage) const;

 private:
  /// exec_unit kind -> set of all memory kinds seen for that exec_unit kind.
  /// SmallPtrSet<1> keeps the common single-element case inline.
  llvm::DenseMap<mlir::Attribute, llvm::SmallPtrSet<mlir::Attribute, 1>>
      exec_to_mem_kinds_;

  void initialize();
};

}  // namespace scheduler::arch_view

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_GROUPLOCALMEMORY_H_
