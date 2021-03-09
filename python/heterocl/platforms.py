import os, subprocess, json, time, sys
from .devices import Platform, CPU, FPGA, PIM, Project
from .tools import *

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
        super(AWS_F1, self).__init__(name, devs, host, xcel, tool)

        self.cache = None
        self.tool = tool
        
class ZC706(Platform):
    def __init__(self):
        name = "zc706"
        devs = [
            CPU("arm", "a9"), 
            FPGA("xilinx", "xc7z045")
        ]
        host = devs[0].set_backend("vhls")
        xcel = devs[1].set_backend("vhls")
        tool = Tool.vivado_hls
        super(ZC706, self).__init__(name, devs, host, xcel, tool)

class VLAB(Platform):
    def __init__(self):
        name = "vlab"
        devs = [
            CPU("intel", "e5"), 
            FPGA("intel", "arria10")
            ]
        host = devs[0].set_backend("aocl")
        xcel = devs[1].set_backend("aocl")
        tool = Tool.aocl
        super(VLAB, self).__init__(name, devs, host, xcel, tool)

Platform.aws_f1  = AWS_F1()
Platform.zc706   = ZC706()
Platform.vlab    = VLAB()