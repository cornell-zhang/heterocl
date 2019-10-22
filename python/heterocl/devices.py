"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError

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
            host = CPU("x86", compiler="aocl", lang="opencl")
            xcel = FPGA("xilinx", compiler="vhls", lang="hlsc")
        elif key == "zynq":
            host = CPU("arm")
            xcel = FPGA("xilinx")
        elif key == "ppac":
            host = CPU("riscv")
            xcel = PIM("ppac")
        else: # unsupported device
            raise DeviceError("not supported")
        tool = Tooling(key, host, xcel)
        return cls(host, xcel, tool)
           
class env(metaclass=platform):
    mode = "sim"
    def __init__(self, host, xcel, tool):
        self.host = host
        self.xcel = xcel
        self.tool = tool

    def __getattr__(self, key):
        return self.tool.__getattr__(key)
   
    def __call__(self, host=None, xcel=None, tool=None):
        if host: 
            assert isinstance(host, Device)
            self.host = host
        if xcel: 
            assert isinstance(xcel, Device)
            self.xcel = xcel
        if tool: 
            assert isinstance(tool, Tooling)
            self.tool = tool

    def __str__(self):
        return str(self.host) + " : " + \
               str(self.xcel)

    def __repr__(self):
        return str(self.host) + " : " + \
               str(self.xcel)

class device(type):
    def __getattr__(cls, key):
        if key == "host":
           return CPU("x86")
        elif key == "xcel":
           return FPGA("xilinx")
        else: # unsupported device
           raise DeviceError("not supported")

class dev(metaclass=device):
    pass

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
    def __init__(self, platform, host, xcel):
        self.platform = platform
        self.mode = "sim"
        self.host = host
        self.xcel = xcel
        self.mapping = {}
        self.mapping["sim"] = { "type" : "csim", 
                                "emulator" : "vivado_hls",
                                "options" : ""}
        self.mapping["impl"] = { "compile"  : "quartus",
                                 "callback" : ""}

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
    def __init__(self, types, model, **kwargs):
        self.types = types
        self.model = model
        self.impls = {"lang": "",
                      "compiler" : ""}
        for key, value in kwargs.items(): 
            self.impls[key] = value

    def __getattr__(self, key):
        return self.impls[key] 

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, model, **kwargs):
        if model not in ["riscv", "arm", "x86", "sparc", "powerpc"]: 
            raise DeviceError(model + " not supported yet")
        super(CPU, self).__init__("CPU", model, **kwargs)
    def __repr__(self):
        return "CPU (" + str(self.model) + ")"

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, model, **kwargs):
        if model not in ["xilinx", "intel"]: 
            raise DeviceError(model + " not supported yet")
        super(FPGA, self).__init__("FPGA", model, **kwargs)
    def __repr__(self):
        return "FPGA (" + str(self.model) + ")"

class GPU(Device):
    """gpu device with different models"""
    def __init__(self, model, **kwargs):
        if model not in ["cuda", "rocm"]: 
            raise DeviceError(model + " not supported yet")
        super(GPU, self).__init__("GPU", model, **kwargs)
    def __repr__(self):
        return "GPU (" + str(self.model) + ")"

class PIM(Device):
    """cpu device with different models"""
    def __init__(self, model, **kwargs):
        if model not in ["ppac"]: 
            raise DeviceError(model + " not supported yet")
        super(CPU, self).__init__("PIM", model, **kwargs)
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

