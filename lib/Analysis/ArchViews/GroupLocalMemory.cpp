//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"

#include <llvm/ADT/SmallPtrSet.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace scheduler::arch_view;

GroupLocalMemory::GroupLocalMemory(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  initialize();
}

void GroupLocalMemory::initialize() {
  // Walk every group in the device definition.  For each group, scan its
  // immediate children for exec_unit ops and memory ops.  For each exec_unit
  // kind the set of local memory kinds is intersected across all groups that
  // contain it — only memory kinds present in every such group are retained.
  // Ambiguity (intersection set size > 1) is not diagnosed here; it
  // surfaces as an error in getLocalMemoryKindForStage if queried.
  getDevice().getBodyRegion().walk([&](mlir::ktdf_arch::GroupOp group) {
    llvm::SmallVector<mlir::Attribute> exec_kinds;
    llvm::SmallPtrSet<mlir::Attribute, 1> mem_kinds;

    for (auto& op : group.getRegion().front().getOperations()) {
      if (auto exec_op =
              mlir::dyn_cast<mlir::ktdf_arch::ExecutionUnitOp>(&op)) {
        if (mlir::Attribute k = exec_op.getKind()) exec_kinds.push_back(k);
      } else if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
        if (mlir::Attribute k = mem_op.getKind()) mem_kinds.insert(k);
      }
    }

    // TODO: only record exec_units that have an explicit ktdf_arch.datapath
    // to/from the memory — co-location alone does not imply access.
    for (mlir::Attribute ek : exec_kinds) {
      auto [it, inserted] = exec_to_mem_kinds_.try_emplace(ek, mem_kinds);
      if (!inserted)
        it->second.remove_if(
            [&](mlir::Attribute mk) { return !mem_kinds.count(mk); });
    }

    return mlir::WalkResult::advance();
  });
}

mlir::Attribute GroupLocalMemory::getLocalMemoryKind(
    mlir::Attribute exec_unit_kind) const {
  auto it = exec_to_mem_kinds_.find(exec_unit_kind);
  if (it == exec_to_mem_kinds_.end() || it->second.size() != 1) return nullptr;
  return *it->second.begin();
}

mlir::Attribute GroupLocalMemory::getLocalMemoryKindForStage(
    mlir::ktdf::StageOp stage) const {
  const auto applicable_units = stage.getApplicableUnits();
  if (!applicable_units || applicable_units->size() != 1) {
    stage->emitError("expected exactly one applicable exec_unit on ktdf.stage");
    return {};
  }
  mlir::Attribute exec_kind = applicable_units->getValue().front();
  mlir::Attribute mem_kind = getLocalMemoryKind(exec_kind);
  if (!mem_kind) {
    stage->emitError("no unambiguous local memory found for exec_unit kind '")
        << exec_kind << "' in device description";
    return {};
  }
  return mem_kind;
}
