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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ProgramUnitBuilder.h"

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace scheduler;

namespace {

// Returns true iff any unit operand of `op_units` resolves (via
// scheduler::getUnitResourceType) to `target`. Returns failure if any
// operand is unresolvable.
llvm::FailureOr<bool> anyOperandHasType(mlir::ValueRange op_units,
                                        scheduler::ResourceType target,
                                        mlir::Operation* diag_op) {
  for (mlir::Value u : op_units) {
    auto rt_opt = scheduler::getUnitResourceType(u);
    if (!rt_opt) {
      diag_op->emitError(
          "ktdflowering-to-dfir: could not resolve component type for unit "
          "operand");
      return mlir::failure();
    }
    if (*rt_opt == target) return true;
  }
  return false;
}

mlir::LogicalResult filterAndUnwrapKTDFLoweringOps(
    mlir::dataflow::ProgramUnitOp pu, scheduler::ResourceType target) {
  // Pass 1: classify and collect inner execute_on ops.
  llvm::SmallVector<mlir::ktdf_lowering::ExecuteOnOp, 8> applicable_execs;
  llvm::SmallVector<mlir::ktdf_lowering::ExecuteOnOp, 8> non_applicable_execs;

  mlir::WalkResult wr =
      pu.walk([&](mlir::ktdf_lowering::ExecuteOnOp inner) -> mlir::WalkResult {
        auto applies =
            anyOperandHasType(inner.getUnits(), target, inner.getOperation());
        if (mlir::failed(applies)) return mlir::WalkResult::interrupt();
        if (*applies) {
          applicable_execs.push_back(inner);
        } else {
          non_applicable_execs.push_back(inner);
        }
        return mlir::WalkResult::advance();
      });
  if (wr.wasInterrupted()) return mlir::failure();

  // Unwrap applicable execute_on ops first, in walk order (post-order: inner
  // before outer). Splicing inner contents out before the outer wrapper is
  // unwrapped keeps the inner-collected pointers valid even if the outer
  // wrapper is later erased as non-applicable.
  for (auto exec : applicable_execs) {
    mlir::Block* body = exec.getBodyBlock();
    exec->getBlock()->getOperations().splice(
        mlir::Block::iterator(exec.getOperation()), body->getOperations(),
        body->begin(), body->end());
    exec.erase();
  }

  // Erase non-applicable execute_on ops, post-order (inner before outer), so a
  // child non-applicable is erased before its non-applicable ancestor whose
  // erase would otherwise destroy it via region tear-down.
  for (auto exec : non_applicable_execs) exec.erase();

  // Pass 2: filter signal ops.
  llvm::SmallVector<mlir::ktdf_lowering::SignalOp, 8> signals_to_erase;
  wr = pu.walk([&](mlir::ktdf_lowering::SignalOp sig) -> mlir::WalkResult {
    auto applies =
        anyOperandHasType(sig.getUnits(), target, sig.getOperation());
    if (mlir::failed(applies)) return mlir::WalkResult::interrupt();
    if (!*applies) signals_to_erase.push_back(sig);
    return mlir::WalkResult::advance();
  });
  if (wr.wasInterrupted()) return mlir::failure();

  for (auto sig : signals_to_erase) sig.erase();
  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::buildProgramUnits(
    mlir::func::FuncOp func, llvm::ArrayRef<mlir::Operation*> work_ops,
    const ResourceToUnits& components) {
  if (components.empty()) return mlir::success();

  mlir::Block& entry = func.getBody().front();
  // Insertion point: just before the terminator. New program_units appear at
  // the bottom of the entry block, in iteration order; the original work_ops
  // (which we'll erase below) sit between the prelude and the new
  // program_units until we erase them.
  mlir::OpBuilder builder(func.getContext());
  builder.setInsertionPoint(entry.getTerminator());

  // Pre-create the shared flat-core-id constants (0..max core) before any
  // program_unit so they dominate every unit body. Collect the distinct core
  // ids from all unit operands across components.
  llvm::DenseMap<int64_t, mlir::Value> core_id_consts;
  mlir::OpBuilder const_builder(entry.getTerminator());
  {
    for (auto& kv : components) {
      for (mlir::Value u : kv.second) {
        if (auto gu = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
                u.getDefiningOp())) {
          if (auto ca = gu->getAttrOfType<mlir::IntegerAttr>("core")) {
            int64_t k = ca.getInt();
            if (!core_id_consts.count(k)) {
              core_id_consts[k] = mlir::arith::ConstantIndexOp::create(
                  const_builder, func.getLoc(), k);
            }
          }
        }
      }
    }
  }

  for (auto& kv : components) {
    scheduler::ResourceType ct = kv.first;
    const llvm::SmallVector<mlir::Value, 4>& units = kv.second;

    auto pu = mlir::dataflow::ProgramUnitOp::create(
        builder, func.getLoc(), mlir::ValueRange(units),
        /*bodyBuilder=*/
        [&](mlir::OpBuilder& body_builder, mlir::Location loc,
            mlir::ValueRange /*iterArgs*/) {
          mlir::IRMapping mapping;
          for (mlir::Operation* op : work_ops) {
            body_builder.clone(*op, mapping);
          }
          // Body builder is responsible for the terminator when supplied
          // (see DataflowOps.cpp:108-117).
          mlir::dataflow::ReturnOp::create(body_builder, loc);
        });

    if (mlir::failed(filterAndUnwrapKTDFLoweringOps(pu, ct))) {
      return mlir::failure();
    }
    if (mlir::failed(scheduler::replaceComputeTileIdWithCoreQuery(
            pu, core_id_consts, const_builder))) {
      return mlir::failure();
    }
  }

  // Erase get_compute_tile_id ops that are now unused (their program_unit-body
  // uses were redirected to per-unit core queries).
  llvm::SmallVector<mlir::ktdp::GetComputeTileIdOp> dead_tile_ids;
  func.walk([&](mlir::ktdp::GetComputeTileIdOp tid) {
    if (tid.use_empty()) dead_tile_ids.push_back(tid);
  });
  for (mlir::ktdp::GetComputeTileIdOp tid : dead_tile_ids) tid.erase();

  // Erase the original work ops from the function entry block. Iterate in
  // reverse so consumers go before producers.
  for (auto it = work_ops.rbegin(); it != work_ops.rend(); ++it) {
    (*it)->erase();
  }

  return mlir::success();
}
