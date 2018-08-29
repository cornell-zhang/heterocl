import heterocl as hcl

NUM_FEATURES = 1024
NUM_TRAINING = 4500
LUT_SIZE = 2048
LUTIN_TWIDTH = 12
LUTIN_IWIDTH = 4
STEP_SIZE = 60000

def SgdLR(data, label, theta, lut):

  label_local = hcl.unpack(label, name = "label_local")
  theta_local = hcl.compute((NUM_FEATURES,), lambda x: 0, "theta_local")

  def Sigmoid(exponent):
    ret = hcl.local(0.0, "sigmoid", theta_local.dtype)
    with hcl.if_(exponent > 4.0):
      ret[0] = 1.0
    with hcl.elif_(exponent < -4.0):
      ret[0] = 0.0
    with hcl.else_():
      exponent = hcl.cast(hcl.UFixed(32, 20), exponent)
      with hcl.if_(exponent < 0.0):
        index = hcl.cast(hcl.UInt(12), LUT_SIZE) + hcl.cast(hcl.UInt(12), (exponent << (LUTIN_TWIDTH - LUTIN_IWIDTH)))
        ret[0] = lut[index]
      with hcl.else_():
        index = hcl.cast(hcl.UInt(12), exponent << (LUTIN_TWIDTH - LUTIN_IWIDTH))
        ret[0] = lut[index]
    return ret[0]

  with hcl.stage("M") as M:
    with hcl.for_(0, NUM_TRAINING) as training_id:
      training_label = hcl.local(label_local[training_id], "training_label", label_local.dtype)
      training_data = hcl.compute((NUM_FEATURES / 4,), lambda x: data[training_id * NUM_FEATURES / 4 + x], "training_data")
      training_instance = hcl.unpack(training_data, name = "training_instance")

      # Main Computation
      k = hcl.reduce_axis(0, NUM_FEATURES, "k")
      dot = hcl.compute((1,), lambda x: hcl.sum(theta_local[k] * training_instance[k], axis = k, dtype = theta_local.dtype), "dot", dtype = theta_local.dtype)
      prob = hcl.compute(dot.shape, lambda x: Sigmoid(dot[x]), "prob", theta_local.dtype)

      gradient = hcl.compute((NUM_FEATURES,), lambda x: (prob[0] - training_label[0]) * training_instance[x], "gradient", dtype = theta_local.dtype)
      update = hcl.update(theta_local, lambda x: theta_local[x] - hcl.cast(gradient.dtype, STEP_SIZE) * gradient[x], "update_param")

  theta_pack = hcl.pack(theta_local, name = "theta_pack", dtype = theta.dtype)
  stream_out = hcl.update(theta, lambda x: theta_pack[x], name = "stream_out")

  return stream_out

DTYPE = hcl.Fixed(16, 4)
LTYPE = hcl.UInt(8)
FTYPE = hcl.Fixed(32, 13)

D_VECTOR_SIZE = 64 / DTYPE.bits
L_VECTOR_SIZE = 32 / LTYPE.bits
F_VECTOR_SIZE = 64 / FTYPE.bits

data = hcl.placeholder((NUM_FEATURES * NUM_TRAINING / D_VECTOR_SIZE,), "data", dtype = hcl.UInt(64))
label = hcl.placeholder((NUM_TRAINING / L_VECTOR_SIZE,), "label", dtype = hcl.UInt(32))
theta = hcl.placeholder((NUM_FEATURES / F_VECTOR_SIZE,), "theta", dtype = hcl.UInt(64))
lut = hcl.placeholder((LUT_SIZE,), "lut", dtype = FTYPE)

f = hcl.make_scheme([data, label, theta, lut], SgdLR)
f.downsize(SgdLR.label_local, LTYPE)
f.quantize(SgdLR.theta_local, FTYPE)
f.quantize(SgdLR.M.training_instance, DTYPE)

#s = hcl.make_schedule([data, label, theta, lut], SgdLR)
s = hcl.make_schedule_from_scheme(f)
print hcl.lower(s, [data, label, theta, lut])
