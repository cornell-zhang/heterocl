module {
  func @two_mm( %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>, %D: tensor<?x?xf32>) -> tensor<?x?xf32>
    {
      // D = \alpha*A*B*C + \beta*D

      // Algorithm
      %l1 = hcl.create_loop_handle "x" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "y" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "r" : !hcl.LoopHandle

      %l4 = hcl.create_loop_handle "x2" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "y1" : !hcl.LoopHandle
      %l6 = hcl.create_loop_handle "k"  : !hcl.LoopHandle

      %l7 = hcl.create_loop_handle "x4" : !hcl.LoopHandle
      %l8 = hcl.create_loop_handle "y2" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
      %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
      %s3 = hcl.create_stage_handle "s3" : !hcl.StageHandle
 
      %alpha = constant 1.5 : f32
      %beta = constant 1.2 : f32
     
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
          %a1 = constant 0.0 : f32
          %sum = affine.for %r = 0 to 2200 iter_args(%sum_iter = %a1) -> (f32) {
            %a = tensor.extract %A[%x, %r] : tensor<?x?xf32> 
            %b = tensor.extract %B[%r, %y] : tensor<?x?xf32> 
            %prod1 = mulf %a, %b : f32
            %sum_next = addf %prod1, %sum_iter : f32
            affine.yield %sum_next : f32
          } { loop_name = "r"}
          %0 = tensor.insert %sum into %out_AB[%x, %y] : tensor<?x?xf32>
        } { loop_name = "y" }
      } { loop_name = "x", stage_name = "s1" }
      
      %m2 = constant 1600 : index
      %n2 = constant 2400 : index

      %out_ABC = tensor.generate %m2, %n2 {
      ^bb0(%i : index, %j : index):
        %elem = constant 0.0 : f32
        tensor.yield %elem : f32
      } : tensor<?x?xf32>

      affine.for %x2 = 0 to 1600 {
        affine.for %y1 = 0 to 2400 {
          %a1 = constant 0.0 : f32
          %sum = affine.for %k = 0 to 1800 iter_args(%sum_iter = %a1) -> (f32) {
            %out_ab = tensor.extract %out_AB[%x2, %k] : tensor<?x?xf32> 
            %c =  tensor.extract %C[%k, %y1] : tensor<?x?xf32> 
            %prod2 = mulf %out_ab, %c : f32
            %sum_next = addf %prod2, %sum_iter : f32
            affine.yield %sum_next : f32
          } { loop_name = "k" }
          %1 = tensor.insert %sum into %out_ABC[%x2, %y1] : tensor<?x?xf32>
        } { loop_name = "y1" }
      } { loop_name = "x2", stage_name = "s2" }
      
      affine.for %x4 = 0 to 1600 {
        affine.for %y2 = 0 to 2400 {
          %out_abc = tensor.extract %out_ABC[%x4, %y2] : tensor<?x?xf32> 
          %d = tensor.extract %D[%x4, %y2] : tensor<?x?xf32> 
          %prod3 = mulf %alpha, %out_abc : f32
          %d_ = mulf %beta, %d : f32
          %sum3 = addf %prod3, %d_ : f32
          %2 = tensor.insert %sum3 into %D[%x4, %y2] : tensor<?x?xf32>
        } { loop_name = "y2" }
      } { loop_name = "x4", stage_name = "s3" }

      // HCL primitives
      //%split_factor_x = constant 10.0 : f32
      %l1_outer, %l1_inner = hcl.split (%s1, %l1, 8)
      %l2_outer, %l2_inner = hcl.split (%s2, %l4, 4)
      // Possible bug in the reorder pass
      //hcl.reorder (%l11, %l10: !hcl.LoopHandle<"y.outer">, !hcl.LoopHandle<"x.inner">)
      //hcl.compute_at (%l4: !hcl.LoopHandle<"x2">, %l7: !hcl.LoopHandle<"x4">)
      
      // Terminator
      return %out_ABC : tensor<?x?xf32>
    }
}
