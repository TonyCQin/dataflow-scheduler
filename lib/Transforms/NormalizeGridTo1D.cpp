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
// NormalizeGridTo1D: flatten N-D grids to 1-D and delinearize tile ids.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "normalize-grid-to-1d"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Normalize Grid To 1D pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_NORMALIZEGRIDTO1DPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

// ASSUMPTION (runtime contract): the flat compute-tile id produced by
// ktdp.get_compute_tile_id is numbered ROW-MAJOR over the original grid
// dimensions, i.e. flat = ((c0*d1)+c1)*d2 + ... for grid [d0,d1,d2,...].
// This pass reconstructs the original per-dimension coordinates under that
// convention. If the hardware/runtime instead numbers tiles column-major,
// the reconstructed coordinates -- and every address derived from them --
// will be WRONG with no diagnostic. The get_compute_tile_id op contract does
// not currently pin this ordering; row-major is the conventional default and
// is self-consistent with the grid flattening this pass performs.

// Reconstruct row-major coordinates from a flat index.
// dims is outermost-first. Returns one Value per dim.
//   suffix(i) = product(dims[i+1..]); suffix(last) = 1
//   coord(i)  = (flat divui suffix(i)) remui dims[i]
//   coord(0)  omits the remui (no-op since flat < product(dims)).
llvm::SmallVector<mlir::Value> delinearizeRowMajor(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value flat,
    llvm::ArrayRef<int64_t> dims) {
  const unsigned k = dims.size();
  llvm::SmallVector<int64_t> suffix(k, 1);
  for (int i = static_cast<int>(k) - 2; i >= 0; --i) {
    suffix[i] = suffix[i + 1] * dims[i + 1];
  }

  llvm::SmallVector<mlir::Value> coords;
  coords.reserve(k);
  for (unsigned i = 0; i < k; ++i) {
    mlir::Value v = flat;
    if (suffix[i] != 1) {
      auto div = mlir::arith::ConstantIndexOp::create(builder, loc, suffix[i]);
      v = mlir::arith::DivUIOp::create(builder, loc, v, div);
    }
    if (i != 0) {  // top coord needs no modulo
      auto mod = mlir::arith::ConstantIndexOp::create(builder, loc, dims[i]);
      v = mlir::arith::RemUIOp::create(builder, loc, v, mod);
    }
    coords.push_back(v);
  }
  return coords;
}

struct NormalizeGridTo1DPass
    : public impl::NormalizeGridTo1DPassBase<NormalizeGridTo1DPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module = getOperation();
    auto* ctx = &getContext();
    auto index_ty = mlir::IndexType::get(ctx);

    mlir::WalkResult result =
        module.walk([&](mlir::func::FuncOp func) -> mlir::WalkResult {
          auto grid_attr = func->getAttrOfType<mlir::ArrayAttr>("grid");
          if (!grid_attr) return mlir::WalkResult::advance();

          if (grid_attr.empty()) {
            func.emitError("Grid must have at least one dimension");
            return mlir::WalkResult::interrupt();
          }

          llvm::SmallVector<int64_t> dims;
          dims.reserve(grid_attr.size());
          for (mlir::Attribute a : grid_attr) {
            auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
            if (!ia) {
              func.emitError("Grid size must be integer attribute");
              return mlir::WalkResult::interrupt();
            }
            int64_t d = ia.getInt();
            if (d <= 0) {
              func.emitError("Grid size must be positive (got ") << d << ")";
              return mlir::WalkResult::interrupt();
            }
            dims.push_back(d);
          }

          const unsigned k = dims.size();
          if (k == 1) return mlir::WalkResult::advance();  // already 1-D

          int64_t flat_size = 1;
          for (int64_t d : dims) flat_size *= d;

          // Rewrite grid attribute to [flat_size] (index-typed entry).
          func->setAttr(
              "grid", mlir::ArrayAttr::get(
                          ctx, {mlir::IntegerAttr::get(index_ty, flat_size)}));

          // Rewrite each get_compute_tile_id in this function.
          llvm::SmallVector<mlir::ktdp::GetComputeTileIdOp> tile_ops;
          func.walk([&](mlir::ktdp::GetComputeTileIdOp op) {
            tile_ops.push_back(op);
          });

          for (mlir::ktdp::GetComputeTileIdOp op : tile_ops) {
            if (op.getNumResults() != k) {
              op.emitError("get_compute_tile_id result count (")
                  << op.getNumResults() << ") does not match grid rank (" << k
                  << ")";
              return mlir::WalkResult::interrupt();
            }
            mlir::OpBuilder builder(op);
            auto flat_op = mlir::ktdp::GetComputeTileIdOp::create(
                builder, op.getLoc(), index_ty);
            mlir::Value flat = flat_op.getResult()[0];
            llvm::SmallVector<mlir::Value> coords =
                delinearizeRowMajor(builder, op.getLoc(), flat, dims);
            for (unsigned i = 0; i < k; ++i) {
              op->getResult(i).replaceAllUsesWith(coords[i]);
            }
            op.erase();
          }

          return mlir::WalkResult::advance();
        });

    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createNormalizeGridTo1DPass() {
  return std::make_unique<NormalizeGridTo1DPass>();
}
