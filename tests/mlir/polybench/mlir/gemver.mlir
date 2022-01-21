module {
  func @gemver( %A: tensor<?x?xf32>, %u1: tensor<?xf32>, %u2: tensor<?xf32>, %v1: tensor<?xf32>, %v2: tensor<?xf32>, %x: tensor<?xf32>, %y: tensor<?xf32>, %w: tensor<?xf32>, %z: tensor<?xf32> ) -> tensor<?x?xf32>
    {
      // Algorithm
      %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
      %l4 = hcl.create_loop_handle "j1" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "i2" : !hcl.LoopHandle
      %l6 = hcl.create_loop_handle "i3" : !hcl.LoopHandle
      %l7 = hcl.create_loop_handle "j2" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
      %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
      %s3 = hcl.create_stage_handle "s3" : !hcl.StageHandle
      %s4 = hcl.create_stage_handle "s4" : !hcl.StageHandle

      %alpha = constant 0.1 : f32
      %beta  = constant 0.1 : f32

      affine.for %i = 0 to 4000 {
        affine.for %j = 0 to 4000 {
          %a_i_j = tensor.extract %A[%i, %j] : tensor<?x?xf32>
          %u1_i  = tensor.extract %u1[%i] : tensor<?xf32>
          %v1_i  = tensor.extract %v1[%i] : tensor<?xf32>
          %u2_i  = tensor.extract %u2[%i] : tensor<?xf32>
          %v2_i  = tensor.extract %v2[%i] : tensor<?xf32>
          %prod1 = mulf %u1_i, %v1_i : f32
          %prod2 = mulf %u2_i, %v2_i : f32
          %sum1  = addf %prod1, %prod2 : f32
          %sum2 = addf %a_i_j, %sum1 : f32
          %0 = tensor.insert %sum2 into %A[%i, %j] : tensor<?x?xf32>
        } { loop_name = "j" }
      } { loop_name = "i", stage_name = "s1" }

      affine.for %i1 = 0 to 4000 {
        affine.for %j1 = 0 to 4000 {
          %a_i_j = tensor.extract %A[%i1, %j1] : tensor<?x?xf32>
          %y_j   = tensor.extract %y[%j1] : tensor<?xf32>
          %x_i   = tensor.extract %x[%i1] : tensor<?xf32>
          %prod1 = mulf %a_i_j, %y_j : f32
          %prod2 = mulf %beta, %prod1 : f32
          %sum1  = addf %prod1, %x_i : f32
          %1 = tensor.insert %sum1 into %x[%i1] : tensor<?xf32>
        } { loop_name = "j1" }
      } { loop_name = "i1", stage_name = "s2" }

      affine.for %i2 = 0 to 4000 {
        %x_i = tensor.extract %x[%i2] : tensor<?xf32>
        %z_i = tensor.extract %z[%i2] : tensor<?xf32>
        %sum1  = addf %z_i, %x_i : f32
        %2 = tensor.insert %sum1 into %x[%i2] : tensor<?xf32>
      } { loop_name = "i2", stage_name = "s3" }

      affine.for %i3 = 0 to 4000 {
        affine.for %j2 = 0 to 4000 {
          %a_i_j = tensor.extract %A[%i3, %j2] : tensor<?x?xf32>
          %x_j   = tensor.extract %x[%j2] : tensor<?xf32>
          %w_i   = tensor.extract %w[%i3] : tensor<?xf32>
          %prod1 = mulf %a_i_j, %x_j : f32
          %prod2 = mulf %alpha, %prod1 : f32
          %sum1  = addf %prod1, %w_i : f32
          %1 = tensor.insert %sum1 into %w[%i3] : tensor<?xf32>
        } { loop_name = "j2" }
      } { loop_name = "i3", stage_name = "s4" }
      
      // Terminator
      return %A : tensor<?x?xf32>
    }
}
