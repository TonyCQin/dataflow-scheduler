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

/// Builds the KTDP to schedule IR (KTDFLow) pipeline, stopping before the
/// final KTDFLowToDFIR conversion. Callers that want to inspect the schedule
/// IR before it becomes DataflowIR can call this, insert a dump, then add
/// createKTDFLowToDFIRPass() themselves. buildKTDPToDFIRPipeline calls this
/// internally.
void buildKTDPToScheduleIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

/// Builds the KTDP to DFIR pipeline with the given pass manager.
///
/// Note: this does not emit the DFIR output. Callers that want the resulting
/// DFIR written out should append `createSplitDFIROutputPass()` themselves.
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_PIPELINE_H_
