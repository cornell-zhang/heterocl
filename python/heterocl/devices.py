"""Define HeteroCL device types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
from .debug import DeviceError

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
    def __init__(self, types="CPU", model="x86"):
        self.types = types
        self.model = model

class CPU(Device):
    """cpu device with different models"""
    def __init__(self, model):
        if model not in ["riscv", "arm", "x86", "sparc", "powerpc"]: 
            raise DeviceError(model + " not supported yet")
        super(CPU, self).__init__("CPU", model)
    def __repr__(self):
        return "CPU (" + str(self.model) + ")"

class FPGA(Device):
    """fpga device with different models"""
    def __init__(self, model):
        if model not in ["xilinx", "intel"]: 
            raise DeviceError(model + " not supported yet")
        super(FPGA, self).__init__("FPGA", model)
    def __repr__(self):
        return "FPGA (" + str(self.model) + ")"

class GPU(Device):
    """gpu device with different models"""
    def __init__(self, model):
        if model not in ["cuda", "rocm"]: 
            raise DeviceError(model + " not supported yet")
        super(GPU, self).__init__("GPU", model)
    def __repr__(self):
        return "GPU (" + str(self.model) + ")"

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

