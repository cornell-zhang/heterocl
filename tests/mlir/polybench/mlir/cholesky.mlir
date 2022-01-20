#map0 = affine_map<(d0) -> (d0)>
module {
  func @cholesky( %A: tensor<?x?xf32> ) -> tensor<?x?xf32>
    {
      // A = L. Transpose(L) where A is +ve definite and 
      //     L is a lower Triangular matrix 

      // Algorithm
      %l1 = hcl.create_loop_handle "i" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "j" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "k" : !hcl.LoopHandle
      %l4 = hcl.create_loop_handle "k1" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle

      affine.for %i = 0 to 4000 {
        affine.for %j = 0 to #map0(%i) {
          affine.for %k = 0 to #map0(%j) {
            %a_i_j = tensor.extract %A[%i, %j] : tensor<?x?xf32>
            %a_i_k = tensor.extract %A[%i, %k] : tensor<?x?xf32>
            %a_j_k = tensor.extract %A[%j, %k] : tensor<?x?xf32>
            %prod = mulf %a_i_k, %a_j_k : f32
            %sub = subf %a_i_j, %prod : f32
            %0 = tensor.insert %sub into %A[%i, %j] : tensor<?x?xf32>
          } { loop_name = "k" }
          %a_i_j = tensor.extract %A[%i, %j] : tensor<?x?xf32>
          %a_j_j = tensor.extract %A[%j, %j] : tensor<?x?xf32>
          %div = divf %a_i_j, %a_j_j : f32
          %1 = tensor.insert %div into %A[%i, %j] : tensor<?x?xf32>
        } { loop_name = "j" }
        affine.for %k1 = 0 to #map0(%i) {
          %a_i_i = tensor.extract %A[%i, %i] : tensor<?x?xf32>
          %a_i_k1 = tensor.extract %A[%i, %k1] : tensor<?x?xf32>
          %prod = mulf %a_i_k1, %a_i_k1 : f32
          %sub = subf %a_i_i, %prod : f32
          %2 = tensor.insert %sub into %A[%i, %i] : tensor<?x?xf32>
        } { loop_name = "k1" }
        %a_i_i = tensor.extract %A[%i, %i] : tensor<?x?xf32>
        %sqrt = math.sqrt %a_i_i : f32
        %3 = tensor.insert %sqrt into %A[%i, %i] : tensor<?x?xf32>
      } { loop_name = "i", stage_name = "s1" }
      
      // Terminator
      return %A : tensor<?x?xf32>
    }
}
