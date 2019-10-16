"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError

def map_gen(platform, types, model, mode):
    pass

class platform(type):
    """The platform class for compute environment setups
    
     serves as meta-class for attr getting
     default platform: aws_f1, zynq, ppac

    Parameters
    ----------
    host: str
        Device of device to place data
    model: str
        Model of device to place date
    """
    def __getattr__(cls, key):
        if key == "aws_f1":
           host = CPU("x86", mode=cls.mode)
           device = FPGA("xilinx")
           return cls(host, device)
        elif key == "zynq":
           host = CPU("arm", key)
           device = FPGA("xilinx", key)
           return cls(host, device)
        elif key == "ppac":
           host = CPU("riscv", key)
           device = PIM("ppac")
           return cls(host, device)
        else: # unsupported device
           raise DeviceError("not supported")
           
class env(metaclass=platform):
    mode = "sim"
    def __init__(self, host, device):
        self.host = host
        self.device = device

    def __str__(self):
        return str(self.host) + " : " + \
               str(self.device)

    def __repr__(self):
        return str(self.host) + " : " + \
               str(self.device)


class Tooling(object):
    """The base class for all device tooling

    each device tooling object maintains a stage dict 
    including mapping from stage -> impl/sim tool + options
    stop impl/sim where running into end of stage list

    Parameters
    ----------
    types: str
        Device of device to place data
    model: str
        Model of device to place date
    """
    def __init__(self, types, model, platform, mode):
        self.types = types
        self.model = model
        self.platform = platform
        self.mode = mode
        self.mapping = { "source" : "",
                         "sim"    : "",
                         "impl"   : "" }
        if types == "CPU": # sim = impl
            self.mapping["source"] = { "lang": "opencl",
                                       "compile" : "aocl",
                                       "options" : "" }
            self.mapping["sim"] = { "env" : "sdaccel",
                                    "compile" : "xcpp" }
        if types == "FPGA": 
            self.mapping["source"] = { "lang": "hlsc",
                                       "compile" : "vhls",
                                       "options" : "" }
            self.mapping["sim"] = {}
            self.mapping["co-sim"] = {}
            self.mapping["syn"] = { "compile"  : "vivado_hls",
                                    "callback" : ""}
            self.mapping[""] = {}
        else: # implementation
            pass

    def __getattr__(self, entry):
        return self.mapping[entry] 

    def __str__(self):
        return str(self.platform) + ":" + \
               str(self.model) + "(" + \
               str(self.mode) + ")"

class Device(object):
    """The base class for all device types

    The default data placement is on CPU.

    Parameters
    ----------
    types: str
        Device of device to place data
    model: str
        Model of device to place date
    """
    def __init__(self, types, model, platform, mode):
        self.types = types
        self.model = model
        self.tool = Tooling(types, model, platform, mode)

    def __getattr__(self, key):
        return self.tool.__getattr__(key)

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, model, platform="aws_f1", mode="sim"):
        if model not in ["riscv", "arm", "x86", "sparc", "powerpc"]: 
            raise DeviceError(model + " not supported yet")
        super(CPU, self).__init__("CPU", model, 
                                  platform, mode)
    def __repr__(self):
        return "CPU (" + str(self.model) + ")"

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, model, platform="aws_f1", mode="sim"):
        if model not in ["xilinx", "intel"]: 
            raise DeviceError(model + " not supported yet")
        super(FPGA, self).__init__("FPGA", model,
                                   platform, mode)
    def __repr__(self):
        return "FPGA (" + str(self.model) + ")"

class GPU(Device):
    """gpu device with different models"""
    def __init__(self, model, platform="aws_f1", mode="sim"):
        if model not in ["cuda", "rocm"]: 
            raise DeviceError(model + " not supported yet")
        super(GPU, self).__init__("GPU", model,
                                  platform, mode)
    def __repr__(self):
        return "GPU (" + str(self.model) + ")"

class PIM(Device):
    """cpu device with different models"""
    def __init__(self, model, platform="ppac", mode="sim"):
        if model not in ["ppac"]: 
            raise DeviceError(model + " not supported yet")
        super(CPU, self).__init__("PIM", model,
                                  platform, mode)
    def __repr__(self):
        return "PIM (" + str(self.model) + ")"

def device_to_str(dtype):
    """Convert a device type to string format.

    Parameters
    ----------
    dtype : Device or str
        The device type to be converted

    Returns
    -------
    str
        The converted device type in string format.
    """
    if isinstance(dtype, Device):
        if isinstance(dtype, CPU):
            return "cpu_" + str(dtype.model)
        elif isinstance(dtype, FPGA):
            return "fpga_" + str(dtype.model)
    else:
        if not isinstance(dtype, str):
            raise DeviceError("Unsupported device type format")
        return dtype

def device_to_hcl(dtype):
    """Convert a device type to Heterocl type.

    Parameters
    ----------
    dtype : Device or str
        The device type to be converted

    Returns
    -------
    Device
    """
    if isinstance(dtype, Device):
        return dtype
    elif isinstance(dtype, str):
        device, model = dtype.split("_") 
        if device == "cpu":
            return CPU(model)
        elif device == "gpu":
            return GPU(model)
        elif device == "fpga":
            return FPGA(model)
        else:
            raise DeviceError("Unrecognized device type")
    else:
        raise DeviceError("Unrecognized device type format")

def get_model(dtype):
    """Get the model of a given device type.

    Parameters
    ----------
    dtype : Device or str
        The given device type

    Returns
    -------
    str
    """
    dtype = dtype_to_hcl(dtype)
    return dtype.types, dtype.model

