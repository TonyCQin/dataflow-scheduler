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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "ktdf-to-operand-lowering"

using namespace scheduler;
using ResourceType = mlir::Attribute;

// Lowercase unit-type tag for the emitted dataflow.get_unit `type`/`name`
// strings (DFIR code generation requires lowercase). Compute resource tokens
// are lowercased directly; Spyre memory-space attributes become their
// lowercased kind ("l1"/"ddr"); any other attribute falls back to its
// lowercased printed form so nothing is left uppercase.
static std::string unitTypeTag(ResourceType rt) {
  if (auto ms = mlir::dyn_cast<mlir::ktdp::SpyreMemorySpaceAttr>(rt)) {
    return mlir::ktdp::stringifySpyreMemorySpaceKind(ms.getValue()).lower();
  }
  if (auto s = mlir::dyn_cast<mlir::StringAttr>(rt)) {
    return s.getValue().lower();
  }
  std::string printed;
  llvm::raw_string_ostream os(printed);
  rt.print(os);
  os.flush();
  return llvm::StringRef(printed).lower();
}

mlir::LogicalResult UnitMaterializer::materialize(
    const ComponentClassification& components, int grid_size,
    UnitSSAMap& unit_ssa_map, mlir::OpBuilder& builder) {
  LDBG(1) << "Step 3: Materialize units";

  assert(grid_size > 0 && "Grid size should be at-least greater than zero");
  auto loc = func_.getLoc();

  // Materialize non-parallel component units: 1 per core
  for (auto component : components.non_parallel_components) {
    std::string comp_name = unitTypeTag(component);

    for (int core = 0; core < grid_size; ++core) {
      auto unit_name = "C" + std::to_string(core) + "-" + comp_name;

      // Create actual dataflow.get_unit operation
      auto get_unit_op = mlir::dataflow::GetUnitOp::create(
          builder, loc, mlir::TypeRange{builder.getIndexType()}, unit_name,
          comp_name);
      get_unit_op->setAttr("core", builder.getI32IntegerAttr(core));
      get_unit_op->setAttr("corelet", builder.getI32IntegerAttr(0));
      mlir::Value unit_value = get_unit_op.getUnit();
      unit_ssa_map.non_parallel[std::make_pair(component, core)] = unit_value;

      LDBG(1) << "  Created non-parallel unit for " << comp_name << " core "
              << core;
    }
  }

  // Materialize parallel component units: 1 per corelet per core
  for (auto& [parallel_op, components_in_parallel] :
       components.parallel_components_map) {
    auto parallel = mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_op);
    if (!parallel) continue;

    int num_instances = parallel.getNumInstances();
    assert(num_instances > 0 &&
           "Number of instances must be greater than zero");

    for (auto component : components_in_parallel) {
      std::string comp_name = unitTypeTag(component);

      for (int corelet = 0; corelet < num_instances; ++corelet) {
        for (int core = 0; core < grid_size; ++core) {
          auto unit_name = "C" + std::to_string(core) + "-" + comp_name +
                           "-CL" + std::to_string(corelet);

          // Create actual dataflow.get_unit operation with corelet attribute
          auto get_unit_op = mlir::dataflow::GetUnitOp::create(
              builder, loc, mlir::TypeRange{builder.getIndexType()}, unit_name,
              comp_name);
          get_unit_op->setAttr("core", builder.getI32IntegerAttr(core));
          get_unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet));
          mlir::Value unit_value = get_unit_op.getUnit();
          unit_ssa_map.parallel[std::make_tuple(parallel_op, component, corelet,
                                                core)] = unit_value;

          LDBG(1) << "  Created parallel unit for " << comp_name << " corelet "
                  << corelet << " core " << core;
        }
      }
    }
  }

  LDBG(1) << "Unit materialization complete";
  return mlir::success();
}

mlir::LogicalResult UnitMaterializer::materializeMemoryUnits(
    const llvm::SetVector<ResourceType>& needed_spaces, int grid_size,
    const scheduler::arch_view::MemoryTree& memory_tree,
    MemoryUnitSSAMap& memory_unit_ssa, mlir::OpBuilder& builder) {
  LDBG(1) << "Materializing memory units";
  auto loc = func_.getLoc();

  for (auto mspace_attr : needed_spaces) {
    // Get string name from the attribute by printing it
    std::string space_name;
    llvm::raw_string_ostream os(space_name);
    mspace_attr.print(os);
    os.flush();
    std::string type_tag = unitTypeTag(mspace_attr);

    if (memory_tree.isGlobalMemory(mspace_attr)) {
      auto get_unit_op = mlir::dataflow::GetUnitOp::create(
          builder, loc, mlir::TypeRange{builder.getIndexType()}, type_tag,
          type_tag);
      memory_unit_ssa[{mspace_attr, -1}] = get_unit_op.getUnit();
      LDBG(1) << "  Created global memory unit for " << space_name;
    } else if (memory_tree.isPerCoreScratchPadMemory(mspace_attr)) {
      for (int core = 0; core < grid_size; ++core) {
        auto unit_name = "C" + std::to_string(core) + "-" + type_tag;
        auto get_unit_op = mlir::dataflow::GetUnitOp::create(
            builder, loc, mlir::TypeRange{builder.getIndexType()}, unit_name,
            type_tag);
        get_unit_op->setAttr("core", builder.getI32IntegerAttr(core));
        memory_unit_ssa[{mspace_attr, core}] = get_unit_op.getUnit();
        LDBG(1) << "  Created per-core memory unit for " << space_name
                << " core " << core;
      }
    } else {
      return func_.emitError("Unknown memory space classification for: ")
             << space_name;
    }
  }
  return mlir::success();
}
