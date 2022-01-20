module {
  func @doitgen( %A: tensor<?x?x?xf32>, %x: tensor<?x?xf32>) -> tensor<?x?x?xf32>
    {
      // A(r, q, p) = \sum_{s = 0}^{S - 1}A(r, q, s) . x(p, s) 

      // Algorithm
      %l1 = hcl.create_loop_handle "r" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "q" : !hcl.LoopHandle
      %l3 = hcl.create_loop_handle "p" : !hcl.LoopHandle
      %l4 = hcl.create_loop_handle "s" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "p1" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle

      // Source: https://llvm.discourse.group/t/creating-a-tensor/4521/3
      %m1 = constant 270 : index

      %sum_ = tensor.generate %m1 {
      ^bb0(%i : index):
        %elem = constant 0.0 : f32
        tensor.yield %elem : f32
      } : tensor<?xf32>

      affine.for %r = 0 to 250 {
        affine.for %q = 0 to 220 {
          affine.for %p = 0 to 270 {
            %temp = constant 0.0 : f32
            %0 = tensor.insert %temp into %sum_[%p] : tensor<?xf32>
            affine.for %s = 0 to 270 {
              %sum__ = tensor.extract %sum_[%p] : tensor<?xf32>
              %a = tensor.extract %A[%r, %q, %s] : tensor<?x?x?xf32>
              %x_ = tensor.extract %x[%s, %p] : tensor<?x?xf32>
              %prod = mulf %a, %x_ : f32
              %sum = addf %sum__, %prod : f32
              %1 = tensor.insert %sum into %sum_[%p] : tensor<?xf32>
            } { loop_name = "s" }
          } { loop_name = "p" }
          affine.for %p1 = 0 to 270 {
            %sum___ = tensor.extract %sum_[%p1] : tensor<?xf32>
            %2 = tensor.insert %sum___ into %A[%r, %q, %p1] : tensor<?x?x?xf32>
          } { loop_name = "p1" }
        } { loop_name = "q" }
      } { loop_name = "r", stage_name = "s1" }

      // Terminator
      return %A : tensor<?x?x?xf32>
    }
}
