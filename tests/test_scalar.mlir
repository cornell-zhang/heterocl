Using mlir as IR
Done HCL-MLIR initialization
module {
  func.func @top() -> memref<1xi32> attributes {itypes = "", otypes = "u"} {
    %0 = memref.alloc() {name = "inst_id", unsigned} : memref<1xi16>
    %c0 = arith.constant 0 : index
    %c2_i16 = arith.constant {unsigned} 2 : i16
    affine.store %c2_i16, %0[0] {to = "inst_id", unsigned} : memref<1xi16>
    %1 = affine.load %0[0] {from = "inst_id", moved, unsigned} : memref<1xi16>
    %2 = memref.alloc() {name = "compute_0", unsigned} : memref<1xi32>
    %3 = hcl.create_op_handle "compute_0"
    %4 = hcl.create_loop_handle %3, "_"
    affine.for %arg0 = 0 to 1 {
      %c0_i32_0 = arith.constant {unsigned} 0 : i32
      affine.store %c0_i32_0, %2[%arg0] {to = "compute_0"} : memref<1xi32>
    } {loop_name = "_", op_name = "compute_0"}
    %c0_i32 = arith.constant {moved} 0 : i32
    %5 = arith.extui %1 {moved} : i16 to i32
    %6 = arith.cmpi eq, %5, %c0_i32 {moved} : i32
    %7 = arith.extui %1 {moved} : i16 to i32
    %c1_i32 = arith.constant {moved} 1 : i32
    %8 = arith.cmpi eq, %7, %c1_i32 {moved} : i32
    scf.if %6 {
      %c0_0 = arith.constant 0 : index
      %c1_i32_1 = arith.constant 1 : i32
      affine.store %c1_i32_1, %2[0] {to = "compute_0", unsigned} : memref<1xi32>
    } else {
      scf.if %8 {
        %c0_0 = arith.constant 0 : index
        %c2_i32 = arith.constant 2 : i32
        affine.store %c2_i32, %2[0] {to = "compute_0", unsigned} : memref<1xi32>
      }
    }
    return %2 : memref<1xi32>
  }
}

