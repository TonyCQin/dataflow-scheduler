// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s


// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
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
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK:             %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK:             %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK:             %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK:             %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK:             ktdf.pipeline {
// CHECK:               %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK:                 %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK:                 %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK:                 %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                 %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                 ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK:               }
// CHECK:               ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK:                 scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:                   %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK:                   %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, %[[CONSTANT_0]] * 64] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["MNILU"]}
// CHECK:               ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:                   %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:                   %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:                   %[[CONSTANT_6:.*]] = arith.constant 255 : index
// CHECK:                   ktdf.pipeline {
// CHECK:                     %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK:                       %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                       %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:                       %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                       %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                       ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK:                     }
// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK:                       %[[CONSTANT_7:.*]] = arith.constant 256 : index
// CHECK:                       scf.for %[[VAL_5:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_7]] step %[[CONSTANT_5]] {
// CHECK:                         %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:                         %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK:                         ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], 0, %[[VAL_5]], 0] size [1, 1, 1, 64] to %[[VAL_4]]#0 size [64] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                       }
// CHECK:                     } {applicable_units = ["L1LU"]}
// CHECK:                     ktdf.stage depends_in(%[[VAL_6:.*]]#2) depends_out(%[[VAL_6]]#3) {
// CHECK:                       %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:                       %[[CONSTANT_8:.*]] = arith.constant 256 : index
// CHECK:                       %[[FOR_0:.*]] = scf.for %[[VAL_7:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_8]] step %[[CONSTANT_5]] iter_args(%[[VAL_8:.*]] = %[[EMPTY_0]]) -> (tensor<1x64xf16>) {
// CHECK:                         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_6]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK:                         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x1x64xf16>) outs(%[[VAL_8]] : tensor<1x64xf16>) {
// CHECK:                         ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:                           %[[ADDF_0:.*]] = arith.addf %[[VAL_9]], %[[VAL_10]] : f16
// CHECK:                           linalg.yield %[[ADDF_0]] : f16
// CHECK:                         } -> tensor<1x64xf16>
// CHECK:                         %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_7]], %[[CONSTANT_6]] : index
// CHECK:                         scf.if %[[CMPI_0]] {
// CHECK:                           ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_6]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK:                         }
// CHECK:                         scf.yield %[[GENERIC_0]] : tensor<1x64xf16>
// CHECK:                       } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK:                     } {applicable_units = ["SFU"]}
// CHECK:                     ktdf.stage depends_in(%[[VAL_11:.*]]#3) depends_out(none) {
// CHECK:                       %[[CONSTANT_9:.*]] = arith.constant 256 : index
// CHECK:                       scf.for %[[VAL_12:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_9]] step %[[CONSTANT_5]] {
// CHECK:                         %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:                         %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK:                         %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_12]], %[[CONSTANT_6]] : index
// CHECK:                         scf.if %[[CMPI_1]] {
// CHECK:                           ktdf.data_transfer from %[[VAL_11]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK:                         }
// CHECK:                       }
// CHECK:                     } {applicable_units = ["L1SU"]}
// CHECK:                   }
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK:               ktdf.stage depends_in(%[[VAL_13:.*]]#3) depends_out(none) {
// CHECK:                 scf.for %[[VAL_14:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:                   %[[SUBI_3:.*]] = arith.subi %[[VAL_14]], %[[CONSTANT_0]] : index
// CHECK:                   %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK:                   ktdf.data_transfer from %[[VAL_13]]#1{{\[}}%[[DIVSI_3]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_14]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               } {applicable_units = ["MNISU"]}
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:         }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
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
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x1x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, %c0 * 64] size [1, 256, 64] to %2#0[%4, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 1, 256, 64] to %3#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
                %5 = tensor.empty() : tensor<1x64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4 : tensor<1x256x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
