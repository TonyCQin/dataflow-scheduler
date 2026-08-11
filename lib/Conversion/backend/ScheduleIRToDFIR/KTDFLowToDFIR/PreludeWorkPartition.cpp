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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/PreludeWorkPartition.h"

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace scheduler;

PreludeWorkSplit scheduler::partitionPreludeAndWork(mlir::func::FuncOp func) {
  PreludeWorkSplit split;
  if (func.getBody().empty()) return split;

  mlir::Block& entry = func.getBody().front();
  for (mlir::Operation& op : entry.without_terminator()) {
    if (mlir::isa<mlir::dataflow::GetUnitOp, mlir::arith::ConstantOp,
                  mlir::uniform::DefImmutableMappingOp,
                  mlir::uniform::QueryMapOp, mlir::ktdp::GetComputeTileIdOp>(
            op)) {
      split.prelude_ops.push_back(&op);
    } else {
      split.work_ops.push_back(&op);
    }
  }
  return split;
}
