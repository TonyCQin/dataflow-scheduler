//===-- Utils.cpp -----------------------------------------------*- c++ -*-===//
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
// This file implements utilities for the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

#include <optional>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"

#define DEBUG_TYPE "ktdf-utilities"

using namespace mlir;
using namespace mlir::ktdf;

void mlir::ktdf::collectStages(Operation* root,
                               SmallVectorImpl<StageOp>& stages) {
  root->walk([&](StageOp stage) { stages.push_back(stage); });
}

void mlir::ktdf::collectQueriedUnits(Operation* root,
                                     SmallVectorImpl<Operation*>& query_ops) {
  root->walk([&](uniform::QueryMapOp op) { query_ops.push_back(op); });
}

auto mlir::ktdf::isTransferTarget(Value memref, Operation* root_op) -> bool {
  if (!root_op) {
    return false;
  }

  return root_op
      ->walk([&](DataTransferOp transfer) {
        // Check if memref is the destination
        if (transfer.getDestination() == memref) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

namespace {

auto isUsedInRegion(Value value, Region& region) -> bool {
  return llvm::any_of(value.getUsers(), [&](Operation* user) -> bool {
    return region.isAncestor(user->getParentRegion());
  });
}

}  // namespace

auto mlir::ktdf::findStageForUnit(uniform::QueryMapOp query_op,
                                  ArrayRef<StageOp> stages) -> StageOp {
  auto unit = query_op->getResult(0);

  for (auto stage : stages) {
    if (isUsedInRegion(unit, stage.getBodyRegion())) {
      return stage;
    }
  }

  return {};
}
