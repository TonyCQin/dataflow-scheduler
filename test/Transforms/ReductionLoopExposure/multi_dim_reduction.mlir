// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s


// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:           func.func @sum_1core() attributes {grid = [1]} {
// CHECK:             call @local_schedule_0() : () -> ()
// CHECK:             return
// CHECK:           }
// CHECK:           func.func private @local_schedule_0()
// CHECK:         }
// CHECK:         ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK:           func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 8589934592 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [4, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x256x64xf16>
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [64], strides: [1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64xf16>
// CHECK:             %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<4x256x64xf16> to memref<4x256x64xf16, "DDR">
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [4, 256, 64], strides: [16384, 64, 1] : memref<4x256x64xf16, "DDR"> to memref<4x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK:             %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<4x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<4x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK:             %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<64xf16> to memref<64xf16, "DDR">
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
// CHECK:             %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK:             ktdf.pipeline {
// CHECK:               %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x2x256x64xf16, "L1">, memref<2x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK:                 %[[ALLOC_0:.*]] = memref.alloc() : memref<2x2x256x64xf16, "L1">
// CHECK:                 %[[ALLOC_1:.*]] = memref.alloc() : memref<2x64xf16, "L1">
// CHECK:                 %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                 %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                 ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x2x256x64xf16, "L1">, memref<2x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK:               }
// CHECK:               ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK:                 scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]] size [2, 256, 64] to %[[VAL_0]]#0{{\[}}%[[VAL_1]], 0, 0, 0] size [1, 2, 256, 64] : memref<4x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x2x256x64xf16, "L1">
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["MNILU"]}
// CHECK:               ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:                   %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:                   %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:                   %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK:                   %[[CONSTANT_7:.*]] = arith.constant 255 : index
// CHECK:                   ktdf.pipeline {
// CHECK:                     %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK:                       %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                       %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:                       %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                       %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                       ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK:                     }
// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK:                       %[[CONSTANT_8:.*]] = arith.constant 2 : index
// CHECK:                       %[[CONSTANT_9:.*]] = arith.constant 256 : index
// CHECK:                       scf.for %[[VAL_5:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_8]] step %[[CONSTANT_5]] {
// CHECK:                         scf.for %[[VAL_6:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_9]] step %[[CONSTANT_5]] {
// CHECK:                           ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_6]], 0] size [1, 1, 1, 64] to %[[VAL_4]]#0 size [64] : memref<2x2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     } {applicable_units = ["L1LU"]}
// CHECK:                     ktdf.stage depends_in(%[[VAL_7:.*]]#2) depends_out(%[[VAL_7]]#3) {
// CHECK:                       %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:                       %[[CONSTANT_10:.*]] = arith.constant 2 : index
// CHECK:                       %[[CONSTANT_11:.*]] = arith.constant 256 : index
// CHECK:                       %[[FOR_0:.*]] = scf.for %[[VAL_8:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_10]] step %[[CONSTANT_5]] iter_args(%[[VAL_9:.*]] = %[[EMPTY_0]]) -> (tensor<64xf16>) {
// CHECK:                         %[[FOR_1:.*]] = scf.for %[[VAL_10:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_11]] step %[[CONSTANT_5]] iter_args(%[[VAL_11:.*]] = %[[VAL_9]]) -> (tensor<64xf16>) {
// CHECK:                           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_7]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK:                           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x1x64xf16>) outs(%[[VAL_11]] : tensor<64xf16>) {
// CHECK:                           ^bb0(%[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16):
// CHECK:                             %[[ADDF_0:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f16
// CHECK:                             linalg.yield %[[ADDF_0]] : f16
// CHECK:                           } -> tensor<64xf16>
// CHECK:                           %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_8]], %[[CONSTANT_6]] : index
// CHECK:                           %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_10]], %[[CONSTANT_7]] : index
// CHECK:                           %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:                           scf.if %[[ANDI_0]] {
// CHECK:                             ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_7]]#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK:                           }
// CHECK:                           scf.yield %[[GENERIC_0]] : tensor<64xf16>
// CHECK:                         }
// CHECK:                         scf.yield %[[FOR_1]] : tensor<64xf16>
// CHECK:                       } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:                     } {applicable_units = ["SFU"]}
// CHECK:                     ktdf.stage depends_in(%[[VAL_14:.*]]#3) depends_out(none) {
// CHECK:                       %[[CONSTANT_12:.*]] = arith.constant 2 : index
// CHECK:                       %[[CONSTANT_13:.*]] = arith.constant 256 : index
// CHECK:                       scf.for %[[VAL_15:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_12]] step %[[CONSTANT_5]] {
// CHECK:                         scf.for %[[VAL_16:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_13]] step %[[CONSTANT_5]] {
// CHECK:                           %[[CMPI_2:.*]] = arith.cmpi eq, %[[VAL_15]], %[[CONSTANT_6]] : index
// CHECK:                           %[[CMPI_3:.*]] = arith.cmpi eq, %[[VAL_16]], %[[CONSTANT_7]] : index
// CHECK:                           %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:                           scf.if %[[ANDI_1]] {
// CHECK:                             ktdf.data_transfer from %[[VAL_14]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[VAL_3]], 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x64xf16, "L1">
// CHECK:                           }
// CHECK:                         }
// CHECK:                       }
// CHECK:                     } {applicable_units = ["L1SU"]}
// CHECK:                   }
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK:               ktdf.stage depends_in(%[[VAL_17:.*]]#3) depends_out(none) {
// CHECK:                 scf.for %[[VAL_18:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:                   ktdf.data_transfer from %[[VAL_17]]#1{{\[}}%[[VAL_18]], 0] size [1, 64] to %[[CAST_1]]{{\[}}%[[CONSTANT_0]]] size [64] : memref<2x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["MNISU"]}
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:         }

// Input tensor in DDR: 4x256x64 (two reduction dims: dim0=2, dim1=256; each
// pipeline iteration processes one 2x256x64 tile).  Two L1 buffer slots hold
// two such tiles so the outer parallel loop runs twice (0..2).
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8589934592 = arith.constant 8589934592 : index
      // DDR input: 4x256x64 (two 2x256x64 tiles processed by the pipeline).
      %0 = ktdp.construct_memory_view %c0, sizes: [4, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<4x256x64xf16> to memref<4x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [4, 256, 64], strides: [16384, 64, 1] : memref<4x256x64xf16, "DDR"> to memref<4x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<4x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<4x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<64xf16> to memref<64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        // Two L1 buffer slots, each holding one 2x256x64 tile.
        %2:4 = ktdf.private -> (memref<2x2x256x64xf16, "L1">, memref<2x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x2x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<2x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<2x2x256x64xf16, "L1">, memref<2x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        // Load stage: two iterations, each loading one 2x256x64 tile from DDR.
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.data_transfer from %cast[%arg0, %c0, %c0] size [2, 256, 64] to %2#0[%arg0, 0, 0, 0] size [1, 2, 256, 64] : memref<4x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x2x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        // Compute stage: two iterations, each reducing one 2x256x64 tile.
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 2, 256, 64] to %3#0 size [32768] : memref<2x2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 32768xf16> -> tensor<2x256x64xf16>
                %5 = tensor.empty() : tensor<64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "parallel"]} ins(%4 : tensor<2x256x64xf16>) outs(%5 : tensor<64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        // Store stage: two iterations, each writing one 64-element result.
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0] size [1, 64] to %cast_2[%c0] size [64] : memref<2x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
