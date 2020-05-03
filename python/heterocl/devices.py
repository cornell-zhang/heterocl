"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError
from .tools import option_table, model_table
from future.utils import with_metaclass

class tooling(type):
    def __getattr__(cls, key):
        if key in option_table:
           return cls(key, *option_table[key])
        else: # unsupported device
           raise DeviceError("not supported")

class tool(with_metaclass(tooling, object)):
    """The base class for all device tooling

    mode (sim/impl) is decided by tool configuration
    e.g. run sw emulation by passing gcc / vivado_hls arg
    and actual impl by passing sdaccel / aocl arg 

    Parameters
    ----------
    types: str
        Device of device to place data
    model: str
        Model of device to place date
    """
    def __init__(self, name, mode, kwargs):
        self.name = name
        self.mode = mode
        self.options = kwargs

    def __getattr__(self, entry):
        return self.mapping[entry] 

    def __call__(self, mode, setting={}):
        self.mode = mode
        self.options = setting
        return self

    def __str__(self):
        return str(self.name) + "-" + \
               str(self.mode) + ":\n" + \
               str(self.options)

    def __repr__(self):
        return str(self.name) + "-" + \
               str(self.mode) + ":\n" + \
               str(self.options)

tool_table = {
  "aws_f1"      : tool("sdaccel",    *option_table["sdaccel"]),
  "zc706"       : tool("vivado_hls", *option_table["vivado_hls"]),
  "ppac"        : tool("rocket",     *option_table["rocket"]),
  "vlab"        : tool("aocl",       *option_table["aocl"]),
  "stratix10_sx": tool("aocl",       *option_table["aocl"]),
  "llvm"        : tool("llvm",       *option_table["llvm"])
}

class Memory(object):
    """The base class for memory modules"""
    def __init__(self, types, cap, channels):
        self.types = types
        self.capacity = cap
        self.channels = channels
        self.port = 0

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("port must be integer")
        if key > self.channels:
            raise DeviceError("port must be within \
                    the channel range %d", self.channels)
        self.port = key
        return self

    def __str__(self):
        return str(self.types) + ":" + \
               str(self.port)

class DRAM(Memory):
    def __init__(self, cap=16, channels=4):
        super(DRAM, self).__init__("DRAM", cap, channels)

class HBM(Memory):
    def __init__(self, cap=32, channels=32):
        super(HBM, self).__init__("HBM", cap, channels)

class PLRAM(Memory):
    def __init__(self, cap=32, channels=32):
        super(PLRAM, self).__init__("PLRAM", cap, channels)

class DevMediaPair(object):
    def __init__(self, dev, media):
        self.xcel = dev
        self.memory  = media

    @property
    def dev(self):
        return self.xcel

    @property
    def media(self):
        return self.memory

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("port must be integer")
        if key > self.media.channels:
            raise DeviceError("port must be within \
                    the channel range %d", self.media.channels)
        self.media.port = key
        return self

    def __str__(self):
        return str(self.xcel) + ":" + \
               str(self.media)

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
    def __init__(self, types, vendor, model, **kwargs):
        self.vendor = vendor
        self.types = types
        self.model = model
        self.impls = { "lang": "" }
        for key, value in kwargs.items(): 
            self.impls[key] = value
        # connect to ddr by default
        self.storage = { "ddr" : DRAM() }

    def __getattr__(self, key):
        """ device hierarchy """
        if key in self.impls.keys():
            return self.impls[key]
        else: # return attached memory
            media = self.storage[key]
            return DevMediaPair(self, media)

    def set_lang(self, lang):
        assert lang in \
            ["xocl", "aocl", "vhls", "ihls", "merlinc", "cuda"], \
            "unsupported lang sepc " + lang
        self.impls["lang"] = lang
        return self


class CPU(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["riscv", "arm", "intel", "sparc", "powerpc"]: 
            raise DeviceError(vendor + " not supported yet")
        assert "cpu_" + model in model_table[vendor], \
            model + " not supported yet"
        super(CPU, self).__init__("CPU", vendor, model, **kwargs)

    def __repr__(self):
        return "cpu-" + self.vendor + "-" + str(self.model) + \
               ":" + self.impls["lang"]

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["xilinx", "intel"]: 
            raise DeviceError(vendor + " not supported yet")
        assert "fpga_" + model in model_table[vendor], \
            model + " not supported yet"
        super(FPGA, self).__init__("FPGA", vendor, model, **kwargs)
        # attach supported memory modules
        if vendor == "xilinx" and "xcvu19p" in model:
            self.storage["hbm"] = HBM()
            self.storage["plram"] = PLRAM()

    def __repr__(self):
        return "fpga-" + self.vendor + "-" + str(self.model) + \
               ":" + self.impls["lang"]

class GPU(Device):
    """gpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["nvidia", "amd"]: 
            raise DeviceError(vendor + " not supported yet")
        assert "gpu_" + model in model_table[vendor], \
            model + " not supported yet"
        super(GPU, self).__init__("GPU", vendor, model, **kwargs)

    def __repr__(self):
        return "gpu-" + self.vendor + "-" + str(self.model) + \
               ":" + self.impls["lang"]

class PIM(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if model not in ["ppac"]: 
            raise DeviceError(model + " not supported yet")
        super(PIM, self).__init__("PIM", vendor, model, **kwargs)
    def __repr__(self):
        return "pim-" + str(self.model)

dev_table = {
  "aws_f1"       : [CPU("intel", "e5"), FPGA("xilinx", "xcvu19p")],
  "vlab"         : [CPU("intel", "e5"), FPGA("intel", "arria10")],
  "zc706"        : [CPU("arm", "a9"), FPGA("xilinx", "xc7z045")],
  "rocc-ppac"    : [CPU("riscv", "riscv"), PIM("ppac", "ppac")],
  "stratix10_sx" : [CPU("arm", "a53"), FPGA("intel", "stratix10_gx")]
}

class env(type):
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
            devs = dev_table[key]
            host = devs[0].set_lang("xocl")
            xcel = devs[1].set_lang("vhls")
        elif key == "zc706":
            devs = dev_table[key]
            host = devs[0].set_lang("vhls")
            xcel = devs[1].set_lang("vhls")
        elif key == "vlab":
            devs = dev_table[key]
            host = devs[0].set_lang("aocl")
            xcel = devs[1].set_lang("aocl")
        elif key == "llvm":
            devs = None 
            host = None 
            xcel = None 
        elif key == "ppac":
            devs = dev_table["rocc-ppac"]
            host = devs[0].set_lang("c")
            xcel = None 
        else: # unsupported device
            raise DeviceError(key + " not supported")
        tool = tool_table[key]
        return cls(key, devs, host, xcel, tool)
           
class platform(with_metaclass(env, object)):
    def __init__(self, name, devs, host, xcel, tool):
        self.name = name
        self.devs = devs
        self.host = host
        self.xcel = xcel
        self.tool = tool

        if isinstance(host, CPU):
            self.cpu = host
        if isinstance(xcel, FPGA):
            self.fpga = xcel
        elif isinstance(xcel, PIM) and xcel.model == "ppac":
            self.ppac = xcel

    def config(self, compile=None, mode=None, backend=None):
        if compile: # check the backend 
          assert compile in option_table.keys(), \
              "not support tool " + compile
          self.tool = tool(compile, *option_table[compile]) 
        
        if mode: # check tool mode 
          modes = ["sw_sim", "hw_sim", "hw_exe", "debug"]
          assert mode in modes, \
              "supported tool mode: " + str(modes)
          self.tool.mode = mode

        if backend: # set up backend lang
          assert backend in ["vhls", "aocl", "sdaccel"], \
              "not support backend lang " + backend
          self.xcel.lang = backend

        # check correctness of device attribute
        if self.host.lang == "":
            self.host.lang = "xocl"

    def __getattr__(self, key):
        """ return tool options """
        return self.tool.__getattr__(key)
   
    def __call__(self, tooling=None):
        if tooling: # check and update
            assert isinstance(tooling, tool)
            self.tool = tooling
        return self

    def __str__(self):
        return str(self.name) + "(" + \
               str(self.host) + " : " + \
               str(self.xcel) + ")"

    def __repr__(self):
        return str(self.name) + "(" + \
               str(self.host) + " : " + \
               str(self.xcel) + ")"

    @classmethod
    def custom(cls, config):
        assert isinstance(config, dict)
        assert "host" in config.keys() 
        if "xcel" not in config.keys():
            print("\33[1;34m[HeteroCL Warning]\33[0m" + "empty xcel slots")
        host = config["host"]
        xcel = None if not "xcel" in config.keys() else config["xcel"]
        devs = [ host ] + xcel
        # TODO: support multiple xcel devs
        if isinstance(xcel, list):
            xcel = xcel[0]
        tool = None
        return cls("custom", devs, host, xcel, tool)


class dev(object):
    def __init__(self, types, vendor, model):
        self.types = types

    @classmethod
    def cpu(cls, vendor, model):
        return CPU(vendor, model)

    @classmethod
    def fpga(cls, vendor, model):
        return FPGA(vendor, model)

    @classmethod
    def gpu(cls, vendor, model):
        return GPU(vendor, model)


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

