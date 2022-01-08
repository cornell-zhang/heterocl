"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .tools import Tool
from .tvm.target import FPGA_TARGETS
from .debug import DSLError, APIError, HCLError, DeviceError

model_table = {
  "fpga"   : {
    "xilinx" : ["xc7z045", "xcvu19p"],
    "intel"  : ["stratix10_gx", "stratix10_dx", "stratix10_mx", "arria10"],
  },

  "cpu"    : {
    "arm"    : ["a7", "a9", "a53"],
    "riscv"  : ["riscv"],
    "intel"  : ["e5", "i7"],
  },
}

dev_mem_map = {
    "DRAM": 0, "HBM": 1, "PLRAM": 2,
    "BRAM": 3, "LUTRAM": 4, "URAM": 5 
}

def is_mem_onchip(mem_type):
    private = False
    assert mem_type in dev_mem_map
    if dev_mem_map[mem_type] > 2:
        private = True
    return private, dev_mem_map[mem_type]

class Memory(object):
    """The base class for memory modules"""
    def __init__(self, types, capacity=0, num_channels=0, channel_id=0):
        # memory device type (e.g., DRAM, HBM)
        self.types = types
        # memory maximum capacity per-bank in GB
        self.capacity = capacity
        # maximum number of memory channels (banks)
        self.num_channels = num_channels
        # channel index to place data
        self.channel_id = channel_id

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("channel_id must be integer")
        if key > self.num_channels:
            raise DeviceError("channel_id must be within \
                    the channel range %d", self.num_channels)
        self.channel_id = key
        return self

    def __str__(self):
        return str(self.types) + ":" + \
               str(self.channel_id)

# Shared memory between host and accelerators
class DRAM(Memory):
    def __init__(self, capacity=16*1024*1024, num_channels=4):
        super(DRAM, self).__init__("DRAM")
        self.capacity = capacity
        self.num_channels = num_channels

class HBM(Memory):
    def __init__(self, capacity=256*1024, num_channels=32):
        super(HBM, self).__init__("HBM")
        self.capacity = capacity 
        self.num_channels = num_channels

class PLRAM(Memory):
    def __init__(self, capacity=32, num_channels=6):
        super(PLRAM, self).__init__("PLRAM")
        self.capacity = capacity
        self.num_channels = num_channels

# Private memory to FPGA device
class BRAM(Memory):
    def __init__(self):
        super(BRAM, self).__init__("BRAM")
        self.port_num = 2

class LUTRAM(Memory):
    def __init__(self):
        super(LUTRAM, self).__init__("LUTRAM")
        self.port_num = 2

class URAM(Memory):
    def __init__(self):
        super(URAM, self).__init__("URAM")
        self.port_num = 2

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
        memory = self.storage[key]
        return DevMemoryPair(self, memory)

    def set_backend(self, backend):
        if backend is None:
            backend = "vhls"
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
    def __init__(self, device, memory):
        self.xcel = device
        self.memory  = memory

    @property
    def dev(self):
        return self.xcel

    @property
    def mem(self):
        return self.memory

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("channel_id must be integer")
        if key > self.memory.num_channels:
            raise DeviceError("channel_id must be within \
                    the channel range %d", self.memory.num_channels)
        self.memory.channel_id = key
        return self

    def __str__(self):
        return f"({self.xcel}, {self.memory}"

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in model_table["cpu"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert model in model_table["cpu"][vendor], model + " not supported yet"
        else:
            model = model_table["cpu"][vendor][0]
        super(CPU, self).__init__("CPU", vendor, model, **kwargs)

    def __repr__(self):
        return f"CPU({self.vendor}, {self.model}, {self.backend}, {self.dev_id})"

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in model_table["fpga"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert model in model_table["fpga"][vendor], "{} not supported yet".format(model)
        else:
            model = model_table["fpga"][vendor][0]
        super(FPGA, self).__init__("FPGA", vendor, model, **kwargs)
    def __repr__(self):
        return f"FPGA({self.vendor}, {self.model}, {self.backend}, {self.dev_id})"

class PIM(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if model not in ["ppac"]: 
            raise DeviceError(model + " not supported yet")
        super(PIM, self).__init__("PIM", vendor, model, **kwargs)
    def __repr__(self):
        return f"PIM({self.model})"

class Project():
    project_name = "project"
    path = "project"
    
class Platform(object):
    def __init__(self, name, devs, host, xcel, tool):
        self.name = name
        self.devs = devs
        self.host = host
        self.xcel = xcel
        self.tool = tool

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
        if compiler:  
            self.tool = getattr(Tool, compiler) 
        
        if compiler == "vivado_hls" and mode is None: # set default mode
            mode = "csim"

        if script is not None: # custom script
            # need to be context string instead of file path
            self.tool.script = script
            mode = "custom"
        else:
            self.tool.script = ""

        if mode is not None: 
            self.tool.set_mode(mode)

        self.xcel.set_backend(backend)
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
        return f"{self.name}({self.host}, {self.xcel})"

    def __repr__(self):
        return f"{self.name}({self.host}, {self.xcel})"

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
        device = FPGA(vendor, model)
        if vendor == "xilinx" and "xcvu19p" in model:
            device.storage["HBM"] = HBM()
        return device


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

