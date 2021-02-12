"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError
from .tools import option_table, model_table
from future.utils import with_metaclass
from .tvm.target import FPGA_TARGETS

dev_mem_map = {
    "DRAM": 0, "HBM": 1, "PLRAM": 2,
    "BRAM": 3, "LUTRAM": 4, "URAM": 5 
}

class dev_mem_type(object):
    @staticmethod
    def is_on_chip(mem_type):
        private = False
        assert mem_type in dev_mem_map
        if dev_mem_map[mem_type] > 2:
            private = True
        return private, dev_mem_map[mem_type]

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
    def __init__(self, types, capacity=0, num_channels=0, port=0):
        # memory device type (e.g., DRAM, HBM)
        self.types = types
        # memory maximum capacity per-bank in GB
        self.capacity = capacity
        # maximum number of memory channels (banks)
        self.num_channels = num_channels
        # channel index to place data
        self.port = port

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("port must be integer")
        if key > self.num_channels:
            raise DeviceError("port must be within \
                    the channel range %d", self.num_channels)
        self.port = key
        return self

    def __str__(self):
        return str(self.types) + ":" + \
               str(self.port)

# Shared memory between host and accelerators
class DRAM(Memory):
    def __init__(self, capacity=16, num_channels=4):
        super(DRAM, self).__init__("DRAM", capacity, num_channels)

class HBM(Memory):
    def __init__(self, capacity=32, num_channels=32):
        super(HBM, self).__init__("HBM", capacity, num_channels)

class PLRAM(Memory):
    def __init__(self, capacity=32, num_channels=6):
        super(PLRAM, self).__init__("PLRAM", capacity, num_channels)

# Private memory to FPGA device
class BRAM(Memory):
    def __init__(self):
        super(BRAM, self).__init__("BRAM", port=2)

class LUTRAM(Memory):
    def __init__(self):
        super(LUTRAM, self).__init__("LUTRAM", port=2)

class URAM(Memory):
    def __init__(self):
        super(URAM, self).__init__("URAM", port=2)

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
        self.types  = types
        self.model  = model

        self.dev_id = 0
        self.backend   = ""
        self.config = dict()

        for key, value in kwargs.items(): 
            self.config[key] = value

        # connect to DRAM by default
        self.storage = { "DRAM" : DRAM() }

    def __getattr__(self, key):
        """ device hierarchy """
        if key in self.config.keys():
            return self.config[key]
        else: # return attached memory
            media = self.storage[key]
            return DevMemoryPair(self, media)

    def set_backend(self, backend):
        assert backend in FPGA_TARGETS, "unsupported backend " + backend
        self.backend = backend
        return self

    def get_dev_id(self):
        return self.dev_id

    def set_dev_id(self, dev_id):
        if not isinstance(dev_id, int):
            raise DeviceError("dev_id must be integer")
        self.dev_id = dev_id

class DevMemoryPair(object):
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
        if key > self.media.num_channels:
            raise DeviceError("port must be within \
                    the channel range %d", self.media.num_channels)
        self.media.port = key
        return self

    def __str__(self):
        return str(self.xcel) + ":" + str(self.media)

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in model_table["cpu"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert model in model_table["cpu"][vendor], \
                model + " not supported yet"
        else:
            model = model_table["cpu"][vendor][0]
        super(CPU, self).__init__("CPU", vendor, model, **kwargs)

    def __repr__(self):
        return "cpu-" + self.vendor + "-" + str(self.model) + \
                ":backend-" + self.backend + ":dev-id-" + str(self.dev_id)

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in model_table["fpga"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert model in model_table["fpga"][vendor], \
                "{} not supported yet".format(model)
        else:
            model = model_table["fpga"][vendor][0]
        super(FPGA, self).__init__("FPGA", vendor, model, **kwargs)
    def __repr__(self):
        return "fpga-" + self.vendor + "-" + str(self.model) + \
               ":backend-" + self.backend + ":dev-id-" + str(self.dev_id)

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
            host = devs[0].set_backend("xocl")
            xcel = devs[1].set_backend("vhls")
        elif key == "zc706":
            devs = dev_table[key]
            host = devs[0].set_backend("vhls")
            xcel = devs[1].set_backend("vhls")
        elif key == "vlab":
            devs = dev_table[key]
            host = devs[0].set_backend("aocl")
            xcel = devs[1].set_backend("aocl")
        elif key == "llvm":
            devs = None 
            host = None 
            xcel = None 
        elif key == "ppac":
            devs = dev_table["rocc-ppac"]
            host = devs[0].set_backend("c")
            xcel = None 
        else: # unsupported device
            raise DeviceError(key + " not supported")
        tool = tool_table[key]
        return cls(key, devs, host, xcel, tool)

class Project():
    project_name = "project"
    path = "project"
    
class Platform(with_metaclass(env, object)):

    def __init__(self, name, devs, host, xcel, tool):
        self.name = name
        self.devs = devs
        self.host = host
        self.xcel = xcel
        self.tool = tool
        self.project = "project"

        if isinstance(host, CPU):
            self.cpu = host
        if isinstance(xcel, FPGA):
            self.fpga = xcel
        elif isinstance(xcel, PIM) and xcel.model == "ppac":
            self.ppac = xcel

        # attach supported memory modules
        if xcel.vendor == "xilinx" and "xcvu19p" in xcel.model:
            off_chip_mem = {
                "HBM": HBM,
                "PLRAM": PLRAM
            }
            for k, v in off_chip_mem.items():
                self.host.storage[k] = v()
                self.xcel.storage[k] = v()

        on_chip_mem = {
            "URAM": URAM,
            "BRAM": BRAM,
            "LUTRAM": LUTRAM
        }
        for k, v in on_chip_mem.items():
            self.xcel.storage[k] = v()


    def config(self, compiler, mode=None,
                     backend=None, script=None,
                     project=None):
        """Configure the HCL runtime platform.

        Parameters
        ----------
        compiler : str
            EDA compiler name (e.g. vitis, vivado_hls)

        mode : str
            EDA tool mode. We currently support sw_sim(software
            simulation), hw_sim (hardware simulation), hw_exe(hardware
            execution), debug (printing out host and device code)

        backend : str
            To configure the backend code generation. 

        script : str
            Custom TCL scripst for FPGA synthesis

        project : str
            Name of the project folder

        Returns
        -------
        Device

        Examples
        -------
            p = hcl.Platform.aws_f1
            p.config(compiler="vitis", mode="hw_exe")
            # Build function with target platform 
            f = hcl.build(s, p)
        """
        assert compiler in option_table.keys(), \
            "not support tool " + compiler
        self.tool = tool(compiler, *option_table[compiler]) 
        
        if compiler == "vivado_hls" and mode is None: # set default mode
            mode = "csim"

        if script is not None: # custom script
            # need to be context string instead of file path
            self.tool.script = script
            mode = "custom"
        else:
            self.tool.script = ""

        if mode is not None: # check tool mode 
            if compiler == "vivado_hls":
                if mode not in ["custom","debug"]:
                    input_modes = mode.split("|")
                    modes = ["csim", "csyn", "cosim", "impl"]
                    new_modes = []
                    for in_mode in input_modes:
                        assert in_mode in modes, \
                            "supported tool mode: " + str(modes)
                        # check validity, dependency shown below
                        # csim (opt) -\    /- cosim
                        #              |--|
                        #    csyn    -/    \- impl
                        if in_mode in ["cosim","impl"]:
                            new_modes.append("csyn")
                            print("Warning: {} needs to be done before {}, ".format("csyn",in_mode) + \
                                "so {} is added to target mode.".format("csyn"))
                        new_modes.append(in_mode)
                    mode = list(set(new_modes))
                    mode.sort(key=lambda x: modes.index(x))
                    mode = "|".join(mode)
            else:
                modes = ["sw_sim", "hw_sim", "hw_exe", "debug"]
                assert mode in modes, \
                    "supported tool mode: " + str(modes)
            self.tool.mode = mode

        if backend is not None: # set up backend backend
            assert backend in ["vhls", "aocl"], "not support backend " + backend
            self.xcel.backend = backend
        else:   
            if compiler == "vitis":
                self.xcel.backend = "vhls"

        # check correctness of device attribute
        if self.host.backend == "":
            self.host.backend = "xocl"

        if project != None:
            Project.project_name = project
            Project.path = project
        self.project = Project.project_name

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
        # set up the default xcel device
        if isinstance(xcel, list): 
            for i in range(len(xcel)):
                xcel[i].set_dev_id(i + 1)
            xcel = xcel[0]

        tool = None
        return cls("custom", devs, host, xcel, tool)


# Class to create custom platform
class dev(object):
    def __init__(self, types, vendor, model):
        self.types = types

    @classmethod
    def CPU(cls, vendor, model=None):
        return CPU(vendor, model)

    @classmethod
    def FPGA(cls, vendor, model=None):
        return FPGA(vendor, model)


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

