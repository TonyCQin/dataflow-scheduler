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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"

#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "ktdf-to-operand-lowering"

using namespace scheduler;

mlir::LogicalResult UniformInfra::createMapsAndQueries(
    const ComponentClassification& components, int grid_size,
    const UnitSSAMap& unit_ssa_map, QueriedUnitsMap& queried_units,
    UniformMapsStorage& uniform_maps, mlir::OpBuilder& builder) {
  LDBG(1) << "Step 4: Create maps and queries";

  auto loc = func_.getLoc();

  // Create symbolic tile ID using ktdp.get_compute_tile_id
  auto tile_id_op = mlir::ktdp::GetComputeTileIdOp::create(
      builder, loc, builder.getIndexType());
  mlir::Value tile_id = tile_id_op.getResult()[0];

  LDBG(1) << "  Created symbolic tile_id";

  // Create maps and queries for non-parallel components
  for (auto component : components.non_parallel_components) {
    std::string comp_name = getComponentName(component);

    llvm::SmallVector<mlir::Value> keys;
    llvm::SmallVector<mlir::Value> values;

    for (int core = 0; core < grid_size; ++core) {
      keys.push_back(mlir::arith::ConstantIndexOp::create(builder, loc, core));

      auto it = unit_ssa_map.non_parallel.find(std::make_pair(component, core));
      if (it == unit_ssa_map.non_parallel.end()) {
        return func_.emitError("Unit SSA value not found for component ")
               << comp_name << " core " << core;
      }
      values.push_back(it->second);
    }

    // Create actual uniform.def_immutable_mapping
    auto map_op = mlir::uniform::DefImmutableMappingOp::create(
        builder, loc, builder.getIndexType(), keys, values);
    mlir::Value map_value = map_op.getResult();
    uniform_maps.non_parallel[component] = map_value;

    // Create actual uniform.query_map
    auto query_op = mlir::uniform::QueryMapOp::create(
        builder, loc, builder.getIndexType(), map_value, tile_id);
    mlir::Value queried = query_op.getResult();
    queried_units.non_parallel[component] = queried;

    LDBG(1) << "  Created map and query for " << comp_name;
  }

  // Create maps and queries for parallel components
  for (auto& [parallel_op, components_in_parallel] :
       components.parallel_components_map) {
    auto parallel = mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_op);
    if (!parallel) continue;

    int num_instances = parallel.getNumInstances();

    for (auto component : components_in_parallel) {
      std::string comp_name = getComponentName(component);

      for (int corelet = 0; corelet < num_instances; ++corelet) {
        llvm::SmallVector<mlir::Value> keys;
        llvm::SmallVector<mlir::Value> values;

        for (int core = 0; core < grid_size; ++core) {
          keys.push_back(
              mlir::arith::ConstantIndexOp::create(builder, loc, core));

          auto it = unit_ssa_map.parallel.find(
              std::make_tuple(parallel_op, component, corelet, core));
          if (it == unit_ssa_map.parallel.end()) {
            return func_.emitError("Parallel unit SSA value not found");
          }
          values.push_back(it->second);
        }

        // Create actual uniform.def_immutable_mapping
        auto map_op = mlir::uniform::DefImmutableMappingOp::create(
            builder, loc, builder.getIndexType(), keys, values);
        mlir::Value map_value = map_op.getResult();
        uniform_maps.parallel[std::make_pair(
            std::make_pair(parallel_op, component), corelet)] = map_value;

        // Create actual uniform.query_map
        auto query_op = mlir::uniform::QueryMapOp::create(
            builder, loc, builder.getIndexType(), map_value, tile_id);
        mlir::Value queried = query_op.getResult();
        queried_units.parallel[std::make_pair(
            std::make_pair(parallel_op, component), corelet)] = queried;

        LDBG(1) << "  Created map and query for " << comp_name << " corelet "
                << corelet;
      }
    }
  }

  LDBG(1) << "Map and query creation complete";
  return mlir::success();
}

std::string UniformInfra::getComponentName(ResourceType component) {
  if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(component)) {
    return str_attr.getValue().str();
  }
  return "unknown";
}

mlir::LogicalResult UniformInfra::buildMemoryUniformMaps(
    mlir::dataflow::ProgramUnitOp pu,
    const llvm::SetVector<ResourceType>& per_core_spaces,
    const MemoryUnitSSAMap& memory_unit_ssa, const SchedulerExtContext& ext_ctx,
    llvm::DenseMap<ResourceType, mlir::Value>& resolved_units,
    mlir::OpBuilder& builder) {
  LDBG(1) << "Building memory uniform maps";
  auto loc = pu.getLoc();

  // The iterator block argument (%arg0) is the query key.
  mlir::Block& body = pu.getRegion().front();
  mlir::Value iter_arg = body.getArgument(0);

  // The program_unit operands are the keys of the uniform map.
  mlir::ValueRange pu_operands = pu.getUnits();

  for (auto mspace : per_core_spaces) {
    llvm::SmallVector<mlir::Value> keys;
    llvm::SmallVector<mlir::Value> values;

    for (mlir::Value pu_op : pu_operands) {
      auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
          pu_op.getDefiningOp());
      if (!get_unit) {
        return pu.emitError(
            "program_unit operand is not defined by dataflow.get_unit");
      }
      auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
      if (!core_attr) {
        return pu.emitError(
            "dataflow.get_unit operand missing 'core' attribute");
      }
      int core = static_cast<int>(core_attr.getInt());

      auto it = memory_unit_ssa.find({mspace, core});
      if (it == memory_unit_ssa.end()) {
        return pu.emitError("memory unit SSA value not found for core ")
               << core;
      }
      keys.push_back(pu_op);
      values.push_back(it->second);
    }

    auto map_op = mlir::uniform::DefImmutableMappingOp::create(
        builder, loc, builder.getIndexType(), keys, values);
    auto query_op = mlir::uniform::QueryMapOp::create(
        builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);
    resolved_units[mspace] = query_op.getResult();

    LDBG(1) << "  Built memory uniform map for " << mspace;
  }
  return mlir::success();
}

llvm::FailureOr<mlir::Value> UniformInfra::buildSignalQueryMap(
    mlir::Value signal_query_map, mlir::dataflow::ProgramUnitOp program_unit,
    mlir::OpBuilder& builder, mlir::Location loc) {
  // Get the original query_map operation
  auto query_op = signal_query_map.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query_op) {
    return mlir::failure();
  }

  // Get the def_immutable_mapping
  auto def_map_op =
      query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!def_map_op) {
    return mlir::failure();
  }

  // Get the keys and values from the original mapping
  auto original_keys = def_map_op.getKeys();
  auto original_values = def_map_op.getValues();

  if (original_keys.size() != original_values.size()) {
    return mlir::failure();
  }

  // The program_unit operands are the keys for the new mapping
  mlir::ValueRange pu_operands = program_unit.getUnits();

  // Build new mapping: [%cur_unit_core0 -> %other_unit_core0, %cur_unit_core1
  // -> %other_unit_core1, ...]
  llvm::SmallVector<mlir::Value> new_keys;
  llvm::SmallVector<mlir::Value> new_values;

  for (mlir::Value pu_op : pu_operands) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_op.getDefiningOp());
    if (!get_unit) {
      return mlir::failure();
    }
    auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
    if (!core_attr) {
      return mlir::failure();
    }
    int core = static_cast<int>(core_attr.getInt());

    // Find the corresponding value in the original mapping for this core
    bool found = false;
    for (size_t i = 0; i < original_keys.size(); ++i) {
      auto key_const =
          original_keys[i].getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (key_const && key_const.value() == core) {
        new_keys.push_back(pu_op);
        new_values.push_back(original_values[i]);
        found = true;
        break;
      }
    }

    if (!found) {
      return mlir::failure();
    }
  }

  // Create new def_immutable_mapping with program_unit operands as keys
  auto new_map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), new_keys, new_values);

  // The iterator block argument (%arg0) is the query key
  mlir::Block& body = program_unit.getRegion().front();
  mlir::Value iter_arg = body.getArgument(0);

  // Create new query_map with iter_arg as key
  auto new_query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), new_map_op.getResult(), iter_arg);

  return new_query_op.getResult();
}
