#map0 = affine_map<(d0) -> (d0 + 1)>
module {
  func @gramschmidt( %A: tensor<?x?xf32>, %Q: tensor<?x?xf32>, %R: tensor<?x?xf32> ) -> tensor<?x?xf32>
    {
      // QR decomposition
      // A = QR
      // A = M x N matrix 
      // Q = M x N matrix 
      // R = N x N matrix 

      // Algorithm
      %l1 = hcl.create_loop_handle "k" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "i" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "i1" : !hcl.LoopHandle
      %l4 = hcl.create_loop_handle "j" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "i2" : !hcl.LoopHandle
      %l6 = hcl.create_loop_handle "i3" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle

      affine.for %k = 0 to 2600 {
        %a = constant 0.0 : f32
        %scalar0 = affine.for %i = 0 to 2000 iter_args(%sum_iter = %a) -> (f32) {
          %a_i_k = tensor.extract %A[%i, %k] : tensor<?x?xf32>
          %prod = mulf %a_i_k, %a_i_k : f32
          %sum_next = addf %sum_iter, %prod : f32
          affine.yield %sum_next : f32
        } { loop_name = "i" }
        %sqrt = math.sqrt %scalar0 : f32
        %0 = tensor.insert %sqrt into %R[%k, %k]: tensor<?x?xf32>
        affine.for %i1 = 0 to 2000 {
          %a_i1_k = tensor.extract %A[%i1, %k] : tensor<?x?xf32>
          %r_k_k = tensor.extract %R[%k, %k] : tensor<?x?xf32>
          %div = divf %a_i1_k, %r_k_k : f32
          %1 = tensor.insert %div into %Q[%i1, %k] : tensor<?x?xf32>
        } { loop_name = "i1" }
        affine.for %j = #map0(%k) to 2600 {
          %2 = tensor.insert %a into %R[%k, %j] : tensor<?x?xf32>
          affine.for %i2 = 0 to 2000 {
            %r_k_j = tensor.extract %R[%k, %j] : tensor<?x?xf32>
            %q_i2_k = tensor.extract %Q[%i2, %k] : tensor<?x?xf32>
            %a_i2_j = tensor.extract %A[%i2, %j] : tensor<?x?xf32>
            %prod = mulf %q_i2_k, %a_i2_j : f32
            %sum = addf %r_k_j, %prod : f32
            %3 = tensor.insert %sum into %R[%k, %j] : tensor<?x?xf32>
          } { loop_name = "i2" }
          affine.for %i3 = 0 to 2000 {
            %a_i3_j = tensor.extract %A[%i3, %j] : tensor<?x?xf32>
            %q_i3_k = tensor.extract %Q[%i3, %k] : tensor<?x?xf32>
            %r_k_j = tensor.extract %R[%k, %j] : tensor<?x?xf32>
            %prod = mulf %q_i3_k, %r_k_j : f32
            %sum= subf %a_i3_j, %prod : f32
            %4 = tensor.insert %sum into %A[%i3, %j] : tensor<?x?xf32>
          } { loop_name = "i2" }
        } { loop_name = "j" }
      } { loop_name = "k", stage_name = "s1" }

      // Terminator
      return %A : tensor<?x?xf32>
    }
}
