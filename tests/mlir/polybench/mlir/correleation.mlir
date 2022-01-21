module {
  func @correlation( %data: tensor<?x?xf32>, %mean: tensor<?xf32>, %stddev: tensor<?xf32>, %corr: tensor<?x?xf32>) -> tensor<?x?xf32>
    {

      // Algorithm
      %l1 = hcl.create_loop_handle "x" : !hcl.LoopHandle
      %l2 = hcl.create_loop_handle "k" : !hcl.LoopHandle

      %l4 = hcl.create_loop_handle "x2" : !hcl.LoopHandle
      %l5 = hcl.create_loop_handle "m" : !hcl.LoopHandle

      %l7 = hcl.create_loop_handle "n" : !hcl.LoopHandle
      
      %l8 = hcl.create_loop_handle "i" : !hcl.LoopHandle
      %l9 = hcl.create_loop_handle "j" : !hcl.LoopHandle
      %l10 = hcl.create_loop_handle "p" : !hcl.LoopHandle
      
      %l11 = hcl.create_loop_handle "q" : !hcl.LoopHandle
      %l12 = hcl.create_loop_handle "r" : !hcl.LoopHandle

      %s1 = hcl.create_stage_handle "s1" : !hcl.StageHandle
      %s2 = hcl.create_stage_handle "s2" : !hcl.StageHandle
      %s3 = hcl.create_stage_handle "s3" : !hcl.StageHandle
      %s4 = hcl.create_stage_handle "s4" : !hcl.StageHandle
      %s5 = hcl.create_stage_handle "s5" : !hcl.StageHandle
 
      %epsilon = constant 0.01 : f32
      %N = constant 1000.0 : f32
     
      affine.for %x = 0 to 2600 {
        %a1 = constant 0.0 : f32
        %sum = affine.for %k = 0 to 3000 iter_args(%sum_iter = %a1) -> (f32) {
          %data_ = tensor.extract %data[%k, %x] : tensor<?x?xf32>
          %sum_next = addf %data_, %sum_iter : f32
          affine.yield %sum_next : f32
        } { loop_name = "k"}
        %div = divf %sum, %N : f32
        %0 = tensor.insert %div into %mean[%x] : tensor<?xf32>
      } { loop_name = "x", stage_name = "s1" }
      
      affine.for %x2 = 0 to 2600 {
        %a1 = constant 0.0 : f32
        %sum = affine.for %m = 0 to 3000 iter_args(%sum_iter = %a1) -> (f32) {
          %data_ = tensor.extract %data[%m, %x2] : tensor<?x?xf32>
          %mean_ = tensor.extract %mean[%x2] : tensor<?xf32>
          %sum1 = subf %data_, %mean_ : f32
          %prod = mulf %sum1, %sum1 : f32
          %sum_next = addf %prod, %sum_iter : f32
          affine.yield %sum_next : f32
        } { loop_name = "k"}
        %1 = tensor.insert %sum into %stddev[%x2] : tensor<?xf32>
      } { loop_name = "x2", stage_name = "s2" }

      affine.for %n = 0 to 2600 {
        %stddev_ = tensor.extract %stddev[%n] : tensor<?xf32>
        %div = divf %stddev_, %N : f32
        %sqrt_ = math.sqrt %div : f32
        %cond = cmpf "ult", %sqrt_, %epsilon : f32
        %sqrt = scf.if %cond -> (f32) {
          %val = constant 1.0 : f32
          scf.yield %val : f32
        } else {
          scf.yield %sqrt_ : f32
        }
        %2 = tensor.insert %sqrt into %stddev[%n] : tensor<?xf32>
      } { loop_name = "n", stage_name = "s3" }

      %m1 = constant 2600 : index
      %n1 = constant 2600 : index

      %cov = tensor.generate %m1, %n1 {
      ^bb0(%i : index, %j : index):
        %elem = constant 0.0 : f32
        tensor.yield %elem : f32
      } : tensor<?x?xf32>

      affine.for %i = 0 to 2600 {
        affine.for %j = 0 to 2600 {
          %a1 = constant 0.0 : f32
          %sum = affine.for %p = 0 to 3000 iter_args(%sum_iter = %a1) -> (f32) {
            %data_ = tensor.extract %data[%p, %i] : tensor<?x?xf32>
            %mean_ = tensor.extract %mean[%i] : tensor<?xf32>
            %sum   = subf %data_, %mean_ : f32
            %prod  = mulf %sum, %sum : f32
            %sum_next = addf %sum_iter, %prod : f32
            affine.yield %sum_next : f32
          } { loop_name = "p" }
          %div = divf %sum, %N : f32
          %3 = tensor.insert %div into %cov[%i, %j] : tensor<?x?xf32>
        } { loop_name = "j" }
      } { loop_name = "i", stage_name = "s4" }

      affine.for %q = 0 to 2600 {
        affine.for %r = 0 to 2600 {
          %cov_ = tensor.extract %cov[%q, %r] : tensor<?x?xf32>
          %stddev_q = tensor.extract %stddev[%q] : tensor<?xf32>
          %stddev_r = tensor.extract %stddev[%r] : tensor<?xf32>
          %prod = mulf %stddev_q, %stddev_r : f32
          %div  = divf %cov_, %prod : f32
          %4 = tensor.insert %div into %corr[%q, %r] : tensor<?x?xf32>
        } { loop_name = "r" }
      } { loop_name = "q", stage_name = "s5" }

      // Terminator
      return %corr : tensor<?x?xf32>
    }
}
