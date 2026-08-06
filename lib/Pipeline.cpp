//===-- Pipeline.cpp --------------------------------------------*- c++ -*-===//
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
// This file implements the scheduler pipeline registration.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Pipeline.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

using namespace scheduler;

void scheduler::buildKTDPToScheduleIRPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  pm.addPass(createKTIRLegalityCheckPass());
  pm.addPass(createComputeGroupExtractionPass());
  pm.addPass(createConstructThreeStagePipelinePass(scheduler_ctx));
  pm.addPass(createPathExpansionPass(scheduler_ctx));
  pm.addPass(createScalarBroadcastLegalizationPass());
  pm.addPass(createNormalizeSCFForLoopsPass());
  // Canonicalize to get rid of intervening code and single iteration loops
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createTileSCFForLoopsPass());
  // Canonicalize to simplify tiling arith ops
  // (probably not needed for custom tiling regime)
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::ktdf::createStageCoarseningPass());
  pm.addPass(mlir::ktdf::createBroadcastPromotionPass());
  pm.addPass(createDoubleBufferingPass(scheduler_ctx));
  // Parallelizing before tile selection is beneficial because the tile size
  // selection pass would now take parallel instances into account while
  // determining tile sizes.
  pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(mlir::ktdf::createTileSizeSelectionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createAffineMinCanonicalizationPass());
  pm.addPass(mlir::ktdf::createSubsumeLinearizeIndexPass());
  pm.addPass(createAddressAssignmentPass(scheduler_ctx));
}

void scheduler::buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  buildKTDPToScheduleIRPipeline(pm, scheduler_ctx);
  // TODO: position of cross-instance parallelization is TBD
  // pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(createNormalizeGridTo1DPass());
  pm.addPass(createKTDFToKTDFLoweringPass(scheduler_ctx));
  pm.addPass(createKTDFLowToDFIRPass());
}
