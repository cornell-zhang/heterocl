import heterocl as hcl

NUM_FEATURES = 1024
NUM_TRAINING = 4400
NUM_TESTING = 600
LUT_SIZE = 2048

def SgdLR(data, label, theta, lut):

  label_local = hcl.unpack(label, name = "label_local")
  theta_local = hcl.unpack(theta, name = "theta_local")
  data_local = hcl.unpack(data, name = "data_local")

  FTYPE = theta_local.dtype
  def Sigmoid(exponent):
    ret = hcl.scalar(0.0, "sigmoid", FTYPE)
    with hcl.if_(exponent > hcl.cast(FTYPE, 4.0)):
      ret[0] = 1.0
    with hcl.elif_(exponent < hcl.cast(FTYPE, -4.0)):
      ret[0] = 0.0
    with hcl.else_():
      with hcl.if_(exponent < hcl.cast(FTYPE, 0.0)):
        num = hcl.scalar(0, dtype = hcl.UFixed(18, 8))
        num[0][18:0] = exponent[29:11]
        num[0] = ~(num[0] << 8) + 1
        index = 2047.0 - num[0]
        ret[0] = lut[hcl.cast(hcl.Int(32), index)]
      with hcl.else_():
        index = exponent[21:11]
        ret[0] = lut[hcl.cast(hcl.Int(32), index)]
    return ret[0]

  with hcl.stage("M"):
    with hcl.for_(0, NUM_TRAINING) as train_id:
      training_instance = hcl.compute((NUM_FEATURES,),
          lambda x: data_local[train_id * NUM_FEATURES + x], "training_instance", data_local.dtype)

      # Main Computation
      k = hcl.reduce_axis(0, NUM_FEATURES, "k")
      dot = hcl.compute((1,),
          lambda x: hcl.sum(theta_local[k] * training_instance[k], axis = k, dtype = FTYPE), "dot", dtype = FTYPE)
      gradient = hcl.compute((NUM_FEATURES,),
          lambda x: (Sigmoid(dot[0]) - label_local[train_id]) * training_instance[x], "gradient", dtype = FTYPE)
      update = hcl.update(theta_local,
          lambda x: theta_local[x] - 2565.0 * gradient[x], name = "update")

  theta_pack = hcl.pack(theta_local, name = "theta_pack", dtype = theta.dtype)
  stream_out = hcl.update(theta, lambda x: theta_pack[x], name = "stream_out")

  return stream_out


