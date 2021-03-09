"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError
from .debug import DSLError, APIError, HCLError

model_table = {
  "xilinx" : ["fpga_xc7z045", "fpga_xcvu19p", "fpga_xcu250"],
  "intel"  : ["cpu_e5", "cpu_i7", "fpga_stratix10_gx", 
              "fpga_stratix10_dx", "fpga_stratix10_mx", "fpga_arria10"],
  "arm"    : ["cpu_a7", "cpu_a9", "cpu_a53"],
  "riscv"  : ["cpu_riscv"]
}

class Tool(object):
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

    def copy_utility(self, path, source):
        raise HCLError("Tool.copy_utility not defined")


class Memory(object):
    """The base class for memory modules"""
    def __init__(self, types, cap=0, channels=0, port=0):
        self.types = types
        self.capacity = cap
        self.channels = channels
        self.port = port

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise DeviceError("port must be integer")
        if key > self.channels:
            raise DeviceError("port must be within \
                    the channel range %d", self.channels)
        self.port = key
        return self

    def __str__(self):
        return str(self.types) + "(" + str(self.port) + ")"

# Shared memory between host and accelerators
class DRAM(Memory):
    def __init__(self, cap=16, channels=4):
        super(DRAM, self).__init__("DRAM", cap, channels)

class HBM(Memory):
    def __init__(self, cap=32, channels=32):
        super(HBM, self).__init__("HBM", cap, channels)

class PLRAM(Memory):
    def __init__(self, cap=32, channels=32):
        super(PLRAM, self).__init__("PLRAM", cap, channels)

class SSD(Memory):
    """Solid state disk connected to host via PCIe"""
    def __init__(self, cap=32, path="/dev/sda"):
        super(SSD, self).__init__("SSD", cap)
        self.path = path

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
        return str(self.xcel) + "." + str(self.media)

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
        self.lang   = ""
        self.config = dict()

        for key, value in kwargs.items(): 
            self.config[key] = value

        # connect to ddr by default
        self.storage = { "DRAM" : DRAM() }

    def __getattr__(self, key):
        """ device hierarchy """
        if key in self.config.keys():
            return self.config[key]
        else: # return attached memory
            media = self.storage[key]
            return DevMediaPair(self, media)

    def set_lang(self, lang):
        assert lang in \
            ["xocl", "aocl", "vhls", "ihls", "merlinc", "cuda"], \
            "unsupported lang sepc " + lang
        self.lang = lang
        return self

    def get_dev_id(self):
        return self.dev_id

    def set_dev_id(self, dev_id):
        if not isinstance(dev_id, int):
            raise DeviceError("dev_id must be integer")
        self.dev_id = dev_id

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["riscv", "arm", "intel", "sparc", "powerpc"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert "cpu_" + model in model_table[vendor], \
                model + " not supported yet"
        else:
            model = model_table[vendor][0]
        super(CPU, self).__init__("CPU", vendor, model, **kwargs)

    def __repr__(self):
        return "cpu-" + self.vendor + "-" + str(self.model) + \
                ":lang-" + self.lang + ":dev-id-" + str(self.dev_id)

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["xilinx", "intel"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert "fpga_" + model in model_table[vendor], \
                "{} not supported yet".format(model)
        else:
            model = model_table[vendor][0]
        super(FPGA, self).__init__("FPGA", vendor, model, **kwargs)
    def __repr__(self):
        return "fpga-" + self.vendor + "-" + str(self.model) + \
               ":lang-" + self.lang + ":dev-id-" + str(self.dev_id)

class GPU(Device):
    """gpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if vendor not in ["nvidia", "amd"]: 
            raise DeviceError(vendor + " not supported yet")
        if model is not None:
            assert "gpu_" + model in model_table[vendor], \
                model + " not supported yet"
        else:
            model = model_table[vendor][0]
        super(GPU, self).__init__("GPU", vendor, model, **kwargs)

    def __repr__(self):
        return "gpu-" + self.vendor + "-" + str(self.model) + \
               ":lang-" + self.lang + ":dev-id-" + str(self.dev_id)

class PIM(Device):
    """cpu device with different models"""
    def __init__(self, vendor, model, **kwargs):
        if model not in ["ppac"]: 
            raise DeviceError(model + " not supported yet")
        super(PIM, self).__init__("PIM", vendor, model, **kwargs)
    def __repr__(self):
        return "pim-" + str(self.model)

# Save the (static) project information
# This information will be updated and used in runtime
class Project():
    project_name = "project"
    path = "project"
    platfrom = None
    post_proc_list = dict()
    
class Platform(object):
    def __init__(self, name, devs, host, xcel, tool):
        self.name = name
        self.devs = devs
        self.host = host
        self.xcel = xcel
        self.tool = tool

        self.project = "project"
        self.to_codegen = False
        self.to_compile = False
        self.to_execute = False
        self.execute_status = False
        self.execute_arguments = dict()

        if isinstance(host, CPU):
            self.cpu = host
        if isinstance(xcel, FPGA):
            self.fpga = xcel
        elif isinstance(xcel, PIM) and xcel.model == "ppac":
            self.ppac = xcel

        # attach supported memory modules
        if xcel.vendor == "xilinx" and "xcvu19p" in xcel.model:
            self.host.storage["HBM"]   = HBM()
            self.host.storage["PLRAM"] = PLRAM()
            self.xcel.storage["HBM"]   = HBM()
            self.xcel.storage["PLRAM"] = PLRAM()

        # attach on-device memory devices
        self.xcel.storage["URAM"] = URAM()
        self.xcel.storage["BRAM"] = BRAM()
        self.xcel.storage["LUTRAM"] = LUTRAM()

    def config(self, compile=None, mode=None,
                     backend=None, script=None,
                     project=None):
        if compile:  
            self.tool = getattr(Tool, compile) 
        
        if compile == "vivado_hls" and mode == None: # set default mode
            mode = "csim"

        if script: # custom script
            # need to be context string instead of file path
            self.tool.script = script
            mode = "custom"
        else:
            self.tool.script = ""

        if mode: # check tool mode 
            if compile == "vivado_hls":
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

        if backend is not None: # set up backend lang
            assert backend in ["vhls", "aocl"], "not support backend lang " + backend
            self.xcel.lang = backend
        else:   
            if compile == "vitis":
                self.xcel.lang = "vhls"

        # check correctness of device attribute
        if self.host.lang == "":
            self.host.lang = "xocl"

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
        return str(self.name) + "(" + str(self.host) + ", " + \
               str(self.xcel) + ")"

    def __repr__(self):
        return str(self.name) + "(" + str(self.host) + ", " + \
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
    
    # check whether the bitstream has been cached
    def initialize(self):
        raise HCLError("Platform.initialize() undefined")
    
    def copy_utility(self, path):
        raise HCLError("Platform.copy_utility() undefined")

    def compile(self, *args, **kwargs):
        raise HCLError("Platform.compile() undefined")

    def execute(self, *args, **kwargs):
        raise HCLError("Platform.execute() undefined")

class dev(object):
    def __init__(self, types, vendor, model):
        self.types = types

    @classmethod
    def cpu(cls, vendor, model=None):
        return CPU(vendor, model)

    @classmethod
    def fpga(cls, vendor, model=None):
        return FPGA(vendor, model)

    @classmethod
    def asic(cls, vendor, model=None):
        return FPGA(vendor, model)

    @classmethod
    def gpu(cls, vendor, model):
        return GPU(vendor, model)

    @classmethod
    def ssd(cls, capacity, path):
        return SSD(capacity, path)

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

