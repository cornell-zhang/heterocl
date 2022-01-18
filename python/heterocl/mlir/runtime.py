import os
import re
import subprocess
from sys import platform
import time
from mlir import ir
from mlir import runtime as rt
import numpy as np
from ..report import parse_xml

# Copied from https://github.com/llvm/llvm-project/blob/4748cc69314ad1ffd85fd6f0265d64fbbeba4430/mlir/test/Integration/Dialect/SparseTensor/python/test_stress.py#L168
class TypeConverter:
  """Converter between NumPy types and MLIR types."""

  def __init__(self, context: ir.Context):
    # Note 1: these are numpy "scalar types" (i.e., the values of
    # np.sctypeDict) not numpy "dtypes" (i.e., the np.dtype class).
    #
    # Note 2: we must construct the MLIR types in the same context as the
    # types that'll be passed to irtype_to_sctype() or irtype_to_dtype();
    # otherwise, those methods will raise a KeyError.
    types_list = [
      (np.float64, ir.F64Type.get(context=context)),
      (np.float32, ir.F32Type.get(context=context)),
      (np.int64, ir.IntegerType.get_signless(64, context=context)),
      (np.int32, ir.IntegerType.get_signless(32, context=context)),
      (np.int16, ir.IntegerType.get_signless(16, context=context)),
      (np.int8, ir.IntegerType.get_signless(8, context=context)),
    ]
    self._sc2ir = dict(types_list)
    self._ir2sc = dict(( (ir,sc) for sc,ir in types_list ))

  def dtype_to_irtype(self, dtype: np.dtype) -> ir.Type:
    """Returns the MLIR equivalent of a NumPy dtype."""
    try:
      return self.sctype_to_irtype(dtype.type)
    except KeyError as e:
      raise KeyError(f'Unknown dtype: {dtype}') from e

  def sctype_to_irtype(self, sctype) -> ir.Type:
    """Returns the MLIR equivalent of a NumPy scalar type."""
    if sctype in self._sc2ir:
      return self._sc2ir[sctype]
    else:
      raise KeyError(f'Unknown sctype: {sctype}')

  def irtype_to_dtype(self, tp: ir.Type) -> np.dtype:
    """Returns the NumPy dtype equivalent of an MLIR type."""
    return np.dtype(self.irtype_to_sctype(tp))

  def irtype_to_sctype(self, tp: ir.Type):
    """Returns the NumPy scalar-type equivalent of an MLIR type."""
    if tp in self._ir2sc:
      return self._ir2sc[tp]
    else:
      raise KeyError(f'Unknown ir.Type: {tp}')

  def get_RankedTensorType_of_nparray(self, nparray: np.ndarray) -> ir.RankedTensorType:
    """Returns the ir.RankedTensorType of a NumPy array.  Note that NumPy
    arrays can only be converted to/from dense tensors, not sparse tensors."""
    # TODO: handle strides as well?
    return ir.RankedTensorType.get(nparray.shape,
                                   self.dtype_to_irtype(nparray.dtype))


def run_process(cmd, pattern=None, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def copy_build_files(target, script=None):
    # make the project folder and copy files
    os.makedirs(target.project, exist_ok=True)
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/")
    project = target.project
    platform = str(target.tool.name)
    mode = str(target.tool.mode)
    if platform == "vivado_hls":
        os.system("cp " + path + "vivado/* " + project)
        os.system("cp " + path + "harness.mk " + project)
        if mode != "custom":
            removed_mode = ["csyn", "csim", "cosim", "impl"]
            selected_mode = mode.split("|")
            for s_mode in selected_mode:
                removed_mode.remove(s_mode)

            new_tcl = ""
            with open(os.path.join(project, "run.tcl"), "r") as tcl_file:
                for line in tcl_file:
                    if ("csim_design" in line and "csim" in removed_mode) \
                            or ("csynth_design" in line and "csyn" in removed_mode) \
                            or ("cosim_design" in line and "cosim" in removed_mode) \
                            or ("export_design" in line and "impl" in removed_mode):
                        new_tcl += "#" + line
                    else:
                        new_tcl += line
        else:  # custom tcl
            print("Warning: custom Tcl file is used, and target mode becomes invalid.")
            new_tcl = script

        with open(os.path.join(project, "run.tcl"), "w") as tcl_file:
            tcl_file.write(new_tcl)
        return "success"
    else:
        raise RuntimeError("Not implemented")


def execute_fpga_backend(target):
    project = target.project
    platform = str(target.tool.name)
    mode = str(target.tool.mode)
    if platform == "vivado_hls":
        assert os.system("which vivado_hls >> /dev/null") == 0, \
            "cannot find vivado hls on system path"
        ver = run_process("g++ --version", "\d\.\d\.\d")[0].split(".")
        assert int(ver[0]) * 10 + int(ver[1]) >= 48, \
            "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

        cmd = "cd {}; make ".format(project)
        if mode == "csim":
            cmd += "csim"
            out = run_process(cmd + " 2>&1")
            runtime = [k for k in out.split("\n") if "seconds" in k][0]
            print("[{}] Simulation runtime {}".format(
                time.strftime("%H:%M:%S", time.gmtime()), runtime))

        elif "csyn" in mode or mode == "custom":
            cmd += "vivado_hls"
            print("[{}] Begin synthesizing project ...".format(
                time.strftime("%H:%M:%S", time.gmtime())))
            subprocess.Popen(cmd, shell=True).wait()
            if mode != "custom":
                out = parse_xml(project, "Vivado HLS", print_flag=True)

        else:
            raise RuntimeError(
                "{} does not support {} mode".format(platform, mode))
    else:
        raise RuntimeError("Not implemented")

def execute_llvm_backend(execution_engine, name, *argv):
    import ipdb; ipdb.set_trace()
