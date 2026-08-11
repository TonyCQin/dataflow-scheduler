//===-- Pipeline.h ----------------------------------------------*- c++ -*-===//
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
// This file declares the scheduler pipelie.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_PIPELINE_H_
#define DATAFLOW_SCHEDULER_PIPELINE_H_

namespace mlir {

class OpPassManager;

}  // namespace mlir

namespace scheduler {

struct SchedulerExtContext;

/// Builds the KTDP to DFIR pipeline with the given pass manager.
/// Pass --dump-schedule-ir-dir=<dir> to write the ScheduleIR (the final KTDF
/// IR after address assignment) to `<dir>/schedule_ir.mlir` before the
/// KTDFToKTDFLow and KTDFLowToDFIR passes run.
///
/// Note: this does not emit the DFIR output. Callers that want the resulting
/// DFIR written out should append `createSplitDFIROutputPass()` themselves.
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_PIPELINE_H_
