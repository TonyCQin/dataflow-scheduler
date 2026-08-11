//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_

#include <map>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Storage for unit SSA values: (component, core) -> Value for non-parallel,
/// (parallel_op, component, corelet, core) -> Value for parallel
struct UnitSSAMap {
  llvm::DenseMap<std::pair<ResourceType, int>, mlir::Value>
      non_parallel;  // (component resource, core) -> Value
  llvm::DenseMap<std::tuple<mlir::Operation*, ResourceType, int, int>,
                 mlir::Value>
      parallel;  // (parallel_op, component resource, corelet, core) -> Value
};

/// Storage for memory-space unit SSA values.
/// Global spaces (DDR): keyed by (memory_space_attr, -1).
/// Per-core spaces (L1): keyed by (memory_space_attr, core).
using MemoryUnitSSAMap =
    llvm::DenseMap<std::pair<ResourceType, int>, mlir::Value>;

/// Materializes dataflow.get_unit operations for all components and cores
class UnitMaterializer {
 public:
  explicit UnitMaterializer(mlir::func::FuncOp func) : func_(func) {}

  /// Create all unit SSA values at function entry
  mlir::LogicalResult materialize(const ComponentClassification& components,
                                  int grid_size, UnitSSAMap& unit_ssa_map,
                                  mlir::OpBuilder& builder);

  /// Emit dataflow.get_unit ops for memory-space components at func entry.
  /// Global spaces (memory_tree.isGlobalMemory): one unit, key core = -1.
  /// Per-core spaces (memory_tree.isPerCoreScratchPadMemory ||
  /// memory_tree.isBelowScratchPad): one unit per core 0..grid_size-1.
  mlir::LogicalResult materializeMemoryUnits(
      const llvm::SetVector<ResourceType>& needed_spaces, int grid_size,
      const scheduler::arch_view::MemoryTree& memory_tree,
      MemoryUnitSSAMap& memory_unit_ssa, mlir::OpBuilder& builder);

 private:
  mlir::func::FuncOp func_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_UNITMATERIALIZER_H_
