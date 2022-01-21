module {
  func @three_mm( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>, %D: tensor<?x?xf32>, %G: tensor<?x?xf32>) -> tensor<?x?xf32>
    {
      // G = P x T = (A.B) . (C.D) 

      // Algorithm
      %l1 = hcl.create_loop_handle "x" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "y" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "r" : !hcl.LoopHandle

      %l4 = hcl.create_loop_handle "x2" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "y1" : !hcl.LoopHandle
      %l6 = hcl.create_loop_handle "k"  : !hcl.LoopHandle

      %l7 = hcl.create_loop_handle "x4" : !hcl.LoopHandle
      %l8 = hcl.create_loop_handle "y2" : !hcl.LoopHandle
      %l9 = hcl.create_loop_handle "q"  : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
      %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
      %s3 = hcl.create_stage_handle "s3" : !hcl.StageHandle
 
      // Source: https://llvm.discourse.group/t/creating-a-tensor/4521/3
      %m1 = constant 1600 : index
      %n1 = constant 1800 : index

      %out_AB = tensor.generate %m1, %n1 {
      ^bb0(%i : index, %j : index):
        %elem = constant 0.0 : f32
        tensor.yield %elem : f32
      } : tensor<?x?xf32>

      affine.for %x = 0 to 1600 {
        affine.for %y = 0 to 1800 {
          affine.for %r = 0 to 2000 {
            %a = tensor.extract %A[%x, %r] : tensor<?x?xf32> 
            %b = tensor.extract %B[%r, %y] : tensor<?x?xf32> 
            %out_ab = tensor.extract %out_AB[%x, %y] : tensor<?x?xf32> 
            %prod1 = mulf %a, %b : f32
            %sum1 = addf %prod1, %out_ab : f32
            %0 = tensor.insert %sum1 into %out_AB[%x, %y] : tensor<?x?xf32>
          } { loop_name = "r"}
        } { loop_name = "y" }
      } { loop_name = "x", stage_name = "s1" }
      
      %m2 = constant 1800 : index
      %n2 = constant 2200 : index

      %out_CD = tensor.generate %m2, %n2 {
      ^bb0(%i : index, %j : index):
        %elem = constant 0.0 : f32
        tensor.yield %elem : f32
      } : tensor<?x?xf32>

      affine.for %x2 = 0 to 1800 {
        affine.for %y1 = 0 to 2200 {
          affine.for %k = 0 to 2400 {
            %c = tensor.extract %C[%x2, %k] : tensor<?x?xf32> 
            %d = tensor.extract %D[%k, %y1] : tensor<?x?xf32> 
            %out_cd = tensor.extract %out_CD[%x2, %y1] : tensor<?x?xf32> 
            %prod2 = mulf %c, %d : f32
            %sum2 = addf %prod2, %out_cd : f32
            %1 = tensor.insert %sum2 into %out_CD[%x2, %y1] : tensor<?x?xf32>
          } { loop_name = "k" }
        } { loop_name = "y1" }
      } { loop_name = "x2", stage_name = "s2" }
      
      affine.for %x4 = 0 to 1600 {
        affine.for %y2 = 0 to 2200 {
          affine.for %q = 0 to 1800 {
            %out_ab = tensor.extract %out_AB[%x4, %q] : tensor<?x?xf32> 
            %out_cd = tensor.extract %out_CD[%q, %y2] : tensor<?x?xf32> 
            %g = tensor.extract %G[%x4, %y2] : tensor<?x?xf32> 
            %prod3 = mulf %out_ab, %out_cd : f32
            %sum3 = addf %prod3, %g : f32
            %2 = tensor.insert %sum3 into %G[%x4, %y2] : tensor<?x?xf32>
          } {loop_name = "q" }
        } { loop_name = "y2" }
      } { loop_name = "x4", stage_name = "s3" }

      // HCL primitives
      %split_factor_x = constant 10.0 : f32
      %l1_outer, %l1_inner = hcl.split (%s1, %l1, 8)
      %l2_outer, %l2_inner = hcl.split (%s2, %l4, 4)
      // Possible bug in the reorder pass
      // hcl.reorder (%l11, %l10: !hcl.LoopHandle<"y.outer">, !hcl.LoopHandle<"x.inner">)
      //hcl.compute_at (%l4: !hcl.LoopHandle<"x2">, %l7: !hcl.LoopHandle<"x4">)
      
      // Terminator
      return %G : tensor<?x?xf32>
    }
}
