// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @memref_reduction() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 256 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x256x64xf16, "L1">
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_2]], %[[VAL_1]], %[[CONSTANT_2]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_3]]} : memref<1x256x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:         dataflow.send %[[QUERY_MAP_0]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:       %[[CONSTANT_BITSTREAM_0:.*]] = vectorchain.constant_bitstream {value = [0x0]} : vector<1xf16>
// CHECK-NEXT:       %[[SHUFFLE_0:.*]] = vectorchain.shuffle input(%[[CONSTANT_BITSTREAM_0]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[SHUFFLE_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:         %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:         %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_2]]) : index
// CHECK-NEXT:         %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_1]] : vector<64xf16>
// CHECK-NEXT:         %[[VECTOR_LOAD_1:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:         %[[BINARY_0:.*]] = vectorchain.binary %[[RECEIVE_0]], %[[VECTOR_LOAD_1]] {binary_op = #vectorchain<binary_operator add>, op_specific_map = #[[$ATTR_2]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:         agen.vector_store %[[BINARY_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]]} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Verify the memref-semantics linalg.generic reduction lowering path:
//   linalg.fill    -> vectorchain.constant_bitstream + shuffle + agen.vector_store
//   linalg.generic (buffer) -> agen.vector_load (accumulator) +
//                              vectorchain.binary {add} +
//                              agen.vector_store (write back)

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @memref_reduction() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu  = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %sfu0], [%c1 -> %sfu1]) : index
    %u_sfu   = uniform.query_map(map:%map_sfu, key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x256x64xf16, "L1">
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c256 step %c1 {
        ktdf.data_transfer from %alloc_l1[%c0, %i, %c0] size [1, 1, 64] to %fifo size [64] : memref<1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    ktdf_lowering.execute_on %u_sfu {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c256 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%alloc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          %sum = arith.addf %in, %out : f16
          linalg.yield %sum : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }
}
