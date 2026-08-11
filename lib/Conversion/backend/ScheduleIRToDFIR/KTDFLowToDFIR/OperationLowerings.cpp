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
//
/// Operation lowerings for KTDFLowToDFIR pass
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/BufferPhaseLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ParallelLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Agen/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Pattern to lower ktdf.read_from_fifo operations
struct LowerReadFromFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::ReadFromFifoOp> {
  LowerReadFromFifoPattern(mlir::MLIRContext* context,
                           arch_view::ResourceKinds& resource_kinds,
                           const ResourceToUnits& components)
      : OpRewritePattern(context),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::ReadFromFifoOp read_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(read_op.getFifoSlot().getType());

    // Convert result type (tensor or memref) to flattened vector type
    auto vector_type =
        getFlattenedVectorType(read_op.getType(), resource_kinds_);
    if (!vector_type) {
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        read_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      read_op.emitError("read_from_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto queried_unit_result = resolveUnitFromFifoAttr(
        fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        read_op.getLoc(), read_op.getOperation());
    if (mlir::failed(queried_unit_result)) {
      return mlir::failure();
    }
    mlir::Value queried_unit = *queried_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, read_op.getLoc(), vector_type, queried_unit,
        /*dbgName=*/nullptr);

    // Replace the read_from_fifo with the receive operation
    rewriter.replaceOp(read_op, receive_op.getData());

    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

struct LowerWriteToFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  LowerWriteToFifoPattern(mlir::MLIRContext* context,
                          arch_view::ResourceKinds& resource_kinds,
                          const ResourceToUnits& components)
      : OpRewritePattern(context),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::WriteToFifoOp write_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(write_op.getFifoSlot().getType());

    // Convert data type (tensor or vector) to flattened vector type
    auto vector_type =
        getFlattenedVectorType(write_op.getData().getType(), resource_kinds_);
    if (!vector_type) {
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        write_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      write_op.emitError("write_to_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto queried_unit_result = resolveUnitFromFifoAttr(
        fifo_slot_type.getDest(), components_, rewriter, program_unit,
        write_op.getLoc(), write_op.getOperation());
    if (mlir::failed(queried_unit_result)) {
      return mlir::failure();
    }
    mlir::Value queried_unit = *queried_unit_result;

    // If the data operand is a memref buffer), load it into a vector first.
    mlir::Value send_data = write_op.getData();
    if (mlir::isa<mlir::MemRefType>(send_data.getType())) {
      send_data =
          emitVectorLoad(rewriter, write_op.getLoc(), vector_type, send_data);
    }

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, write_op.getLoc(), queried_unit,
                                   send_data, /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the write_to_fifo operation
    rewriter.eraseOp(write_op);

    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

/// Helper to find the current unit's query_map from signal operands
/// Returns the query_map whose first mapping result matches a program_unit
/// operand
static llvm::FailureOr<mlir::Value> findCurrentUnitQueryMap(
    llvm::ArrayRef<mlir::Value> signal_units,
    llvm::ArrayRef<mlir::Value> program_unit_operands, mlir::Operation* op) {
  for (auto signal_unit : signal_units) {
    // Get the query_map operation
    auto query_op = signal_unit.getDefiningOp<mlir::uniform::QueryMapOp>();
    if (!query_op) {
      op->emitError("signal operand must be a uniform.query_map result");
      return mlir::failure();
    }

    // Get the def_immutable_mapping
    auto def_map_op =
        query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    if (!def_map_op) {
      op->emitError("query_map must reference a def_immutable_mapping");
      return mlir::failure();
    }

    // Get the first value from the mapping (first result unit)
    auto values = def_map_op.getValues();
    if (values.empty()) {
      op->emitError("def_immutable_mapping must have at least one value");
      return mlir::failure();
    }
    mlir::Value first_unit = values[0];

    // Check if this first unit is in the current program_unit operands
    for (auto pu_unit : program_unit_operands) {
      if (first_unit == pu_unit) {
        return signal_unit;
      }
    }
  }

  op->emitError(
      "signal must include at least one query_map whose first mapping result "
      "is in the current program_unit");
  return mlir::failure();
}
/// Pattern to lower ktdf.tiling.derive_size operations using conditional
/// branching
struct LowerGetTileSizePattern
    : public mlir::OpRewritePattern<mlir::ktdf::TilingDeriveSizeOp> {
  LowerGetTileSizePattern(mlir::MLIRContext* context,
                          const SchedulerExtContext& /*scheduler_ctx*/,
                          const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::TilingDeriveSizeOp derive_size_op,
      mlir::PatternRewriter& rewriter) const override {
    auto loc = derive_size_op.getLoc();
    auto ivs = derive_size_op.getIvs();
    auto tile_sizes = derive_size_op.getTileSizes();
    auto total_size = derive_size_op.getTotalSize();

    // We only handle the single-level case (one [iv : tile_size] pair)
    if (ivs.size() != 1) {
      return mlir::failure();
    }

    auto iv = ivs[0];
    auto tile_size = tile_sizes[0];

    // Find the enclosing scf.for loop that uses this IV
    auto iv_block_arg = mlir::dyn_cast<mlir::BlockArgument>(iv);
    if (!iv_block_arg) {
      derive_size_op.emitError("IV must be a block argument");
      return mlir::failure();
    }

    auto for_op = iv_block_arg.getOwner()->getParentOp();
    auto scf_for = mlir::dyn_cast<mlir::scf::ForOp>(for_op);

    if (!scf_for) {
      derive_size_op.emitError("IV must be from an scf.for loop");
      return mlir::failure();
    }
    // Extract constant values - both must be constants
    auto tile_size_const =
        tile_size.getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto total_size_const =
        total_size.getDefiningOp<mlir::arith::ConstantIndexOp>();

    if (!tile_size_const || !total_size_const) {
      derive_size_op.emitError("tile_size and total_size must be constants");
      return mlir::failure();
    }

    int64_t tile_size_val = tile_size_const.value();
    int64_t total_size_val = total_size_const.value();

    // Epilogue size is the remainder; zero means total_size divides tile_size
    // evenly and every iteration — including the last — uses the steady-state
    // tile size.
    int64_t epilogue_size_val = total_size_val % tile_size_val;

    mlir::Value result;
    if (epilogue_size_val == 0) {
      // No epilogue: every tile has the steady-state size; no conditional
      // needed.
      result = tile_size;
    } else {
      // Epilogue exists: last iteration gets epilogue_size, others get
      // tile_size.
      auto upper_bound = scf_for.getUpperBound();
      auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto upper_bound_minus_1 =
          mlir::arith::SubIOp::create(rewriter, loc, upper_bound, c1);
      auto is_not_last = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, iv,
          upper_bound_minus_1);
      auto epilogue_size = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, epilogue_size_val);
      result = mlir::arith::SelectOp::create(rewriter, loc, is_not_last,
                                             tile_size, epilogue_size);
    }

    rewriter.replaceOp(derive_size_op, result);

    return mlir::success();
  }
};

/// Pattern to lower ktdf_lowering.signal operations
struct LowerSignalPattern
    : public mlir::OpRewritePattern<mlir::ktdf_lowering::SignalOp> {
  LowerSignalPattern(mlir::MLIRContext* context,
                     const SchedulerExtContext& /*scheduler_ctx*/,
                     const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf_lowering::SignalOp signal_op,
      mlir::PatternRewriter& rewriter) const override {
    // Signal operations must have at least 2 units
    if (signal_op.getNumUnits() < 2) {
      signal_op.emitError(
          "signal operation must have at least 2 unit operands");
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        signal_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      signal_op.emitError("signal must be inside a program_unit");
      return mlir::failure();
    }

    auto signal_units = signal_op.getUnits();
    auto program_unit_operands = program_unit.getUnits();

    // Convert to vectors for the helper function
    llvm::SmallVector<mlir::Value, 8> signal_units_vec(signal_units.begin(),
                                                       signal_units.end());
    llvm::SmallVector<mlir::Value, 8> program_unit_vec(
        program_unit_operands.begin(), program_unit_operands.end());

    // Find which signal operand corresponds to the current program_unit
    auto curr_unit_query_map_result = findCurrentUnitQueryMap(
        signal_units_vec, program_unit_vec, signal_op.getOperation());
    if (mlir::failed(curr_unit_query_map_result)) {
      return mlir::failure();
    }
    mlir::Value curr_unit_query_map = *curr_unit_query_map_result;

    auto curr_resource_type =
        scheduler::getUnitTypeFromQueryMap(curr_unit_query_map);
    assert(!curr_resource_type.empty() && "getUnitTypeFromQueryMap failed");

    // Build query maps for all other units (not the current unit)
    // Only include units with different resource types than the current unit
    llvm::SmallVector<mlir::Value, 8> other_query_maps;
    for (auto signal_unit : signal_units) {
      if (signal_unit != curr_unit_query_map) {
        auto other_resource_type =
            scheduler::getUnitTypeFromQueryMap(signal_unit);
        assert(!other_resource_type.empty() &&
               "getUnitTypeFromQueryMap failed");

        // Only sync with units of different types. This avoids syncing between
        // corelets 0 and 1 of the same unit type for example.
        if (other_resource_type != curr_resource_type) {
          auto other_query_map_result = UniformInfra::buildSignalQueryMap(
              signal_unit, program_unit, rewriter, signal_op.getLoc());
          if (mlir::failed(other_query_map_result)) {
            signal_op.emitError("failed to build query map for signal operand");
            return mlir::failure();
          }
          other_query_maps.push_back(*other_query_map_result);
        }
      }
    }

    // Create all sync_send operations (from current unit to other units)
    bool wait_immediately_for_async_transfer = true;
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncSendOp::create(
          rewriter, signal_op.getLoc(), other_query_map, /*dbgName=*/nullptr,
          rewriter.getBoolAttr(wait_immediately_for_async_transfer));
    }

    // Create all sync_recv operations (from other units to current unit)
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncRecvOp::create(rewriter, signal_op.getLoc(),
                                         other_query_map, /*dbgName=*/nullptr);
    }

    // Erase the signal operation
    rewriter.eraseOp(signal_op);

    return mlir::success();
  }
};

}  // namespace

mlir::LogicalResult scheduler::runOperationLowerings(
    mlir::func::FuncOp func,
    const scheduler::SchedulerExtContext& scheduler_ctx,
    const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  // Lower linalg.generic compute operations and FIFO operations
  mlir::RewritePatternSet patterns(func.getContext());
  populateLinalgLoweringPatterns(patterns, resource_kinds);
  patterns.add<LowerReadFromFifoPattern>(func.getContext(), resource_kinds,
                                         components);
  patterns.add<LowerWriteToFifoPattern>(func.getContext(), resource_kinds,
                                        components);
  populateDataTransferLoweringPatterns(patterns, components, resource_kinds);
  patterns.add<LowerSignalPattern>(func.getContext(), scheduler_ctx,
                                   components);
  patterns.add<LowerGetTileSizePattern>(func.getContext(), scheduler_ctx,
                                        components);
  if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
    return mlir::failure();
  }

  // Clone loops for buffer phase tracking and replace operations
  if (mlir::failed(lowerDoubleBuffering(func, components, resource_kinds))) {
    return mlir::failure();
  }

  // Lower ktdf.parallel operations after all other lowerings are complete
  if (mlir::failed(lowerParallelOps(func))) {
    return mlir::failure();
  }

  return mlir::success();
}

// Made with Bob
