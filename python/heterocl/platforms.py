import os, subprocess, json, time, sys, json
from .devices import Platform, CPU, FPGA, PIM, Project
from .devices import HBM, PLRAM, LUTRAM, BRAM, URAM
from .tools import *

# Traverse JSON tree in recursive DFS
def traverse_json_tree(node):
    assert "type" in node, "invalid device (JSON) object missing 'type'"
    for key, v in node.items():

        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            # Attributes in a device node
            if key == "type":
                # Compute device that can take compute offloading
                if "compute" in v:
                    print(f"[+] device node {v} ({node['part']})")


        elif isinstance(v, list):
            if key == "interfaces":
                # Connect to other super nodes at same level
                pass
            elif key == "interconnects":
                # Instantiate edges for connection between children nodes
                pass
            elif key == "components":
                # Do nothing
                pass

            for elem in v:
                traverse_json_tree(elem)
        else:
            assert isinstance(v, dict), f"illegal dict value found {v}"
            # Stop traversing if the node does not have any sub-nodes
            if "interconnects" in v and "components" in v:
                traverse_json_tree(v)
            

def import_json_platform(json_file):
    assert os.path.exists(json_file), f"{json_file} does not exist"
    with open(json_file, "r+") as f:
        json_spec = json.load(f)
        traverse_json_tree(json_spec) 


class AWS_F1(Platform):
    def __init__(self):
        name = "aws_f1"
        devs = [
            CPU("intel", "e5"), 
            FPGA("xilinx", "xcvu19p")
            ]
        host = devs[0].set_backend("xocl")
        xcel = devs[1].set_backend("vhls")
        tool = Tool.vitis

        self.AMI_ID = "ami-0a7b98fdb062be15f"
        self.XPFM = "xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm"
        self.cache = None
        self.tool = tool

        # attach supported memory modules
        off_chip_mem = {
            "HBM": HBM,
            "PLRAM": PLRAM
        }
        for memory, memory_class in off_chip_mem.items():
            host.storage[memory] = memory_class()
            xcel.storage[memory] = memory_class()

        on_chip_mem = {
            "URAM": URAM,
            "BRAM": BRAM,
            "LUTRAM": LUTRAM
        }
        for memory, memory_class in on_chip_mem.items():
            xcel.storage[memory] = memory_class()
        super(AWS_F1, self).__init__(name, devs, host, xcel, tool)

class XILINX_ZC706(Platform):
    def __init__(self):
        name = "zc706"
        devs = [
            CPU("arm", "a9"), 
            FPGA("xilinx", "xc7z045")
        ]
        host = devs[0].set_backend("vhls")
        xcel = devs[1].set_backend("vhls")
        tool = Tool.vivado_hls
        on_chip_mem = {
            "URAM": URAM,
            "BRAM": BRAM,
            "LUTRAM": LUTRAM
        }
        for memory, memory_class in on_chip_mem.items():
            xcel.storage[memory] = memory_class()
        super(XILINX_ZC706, self).__init__(name, devs, host, xcel, tool)

class INTEL_VLAB(Platform):
    def __init__(self):
        name = "vlab"
        devs = [
            CPU("intel", "e5"), 
            FPGA("intel", "arria10")
            ]
        host = devs[0].set_backend("aocl")
        xcel = devs[1].set_backend("aocl")
        tool = Tool.aocl
        super(INTEL_VLAB, self).__init__(name, devs, host, xcel, tool)

Platform.aws_f1  = AWS_F1()
Platform.xilinx_zc706  = XILINX_ZC706()
Platform.intel_vlab    = INTEL_VLAB()