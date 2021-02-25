import os, subprocess
from .devices import Platform, dev_table, tool_table

def run_shell_script(command):
    with open("temp.sh", "w") as fp:
        fp.write(command)
    ret = subprocess.run(['sh', 'temp.sh'], 
        stdout=subprocess.PIPE, check=True, shell=True)
    return ret.stdout.decode('utf-8')

# define some built-in platforms in HCL
class AWS_F1(Platform):
    def __init__(self):
        name = "aws_f1"
        devs = dev_table[name]
        host = devs[0].set_lang("xocl")
        xcel = devs[1].set_lang("vhls")
        tool = tool_table[name]
        self.AMI_ID = "ami-0a7b98fdb062be15f"
        super(AWS_F1, self).__init__(name, devs, host, xcel, tool)
    
    # check if the bitstream compiled before + clean up
    def initialize(self, dev_hash, path):
        pass

    # upload the work project to AWS
    def upload(self):
        pass

    # register AFI image
    def register(self):
        pass

    def create_instance(self, aws_key, instance):
        command = "aws ec2 run-instances --image-id {} \
            --security-group-ids sg-08111c9462c75f193 \
            --block-device-mapping DeviceName=/dev/sda1,Ebs={VolumeSize=100} \
            --instance-type {} --key-name {};".format(self.AMI_ID, instance, aws_key)
        out = run_shell_script(command)

    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote AWS machines)
    def compile(self, remote=False, aws_key_path=None, instance="t2.micro"):
        if remote:
            assert os.path.exists(aws_key_path)
            aws_key = None
    
    def download(self):
        pass

class ZC706(Platform):
    def __init__(self):
        name = "zc706"
        devs = dev_table[name]
        host = devs[0].set_lang("vhls")
        xcel = devs[1].set_lang("vhls")
        tool = tool_table[name]
        super(ZC706, self).__init__(name, devs, host, xcel, tool)

class VLAB(Platform):
    def __init__(self):
        name = "vlab"
        devs = dev_table[name]
        host = devs[0].set_lang("aocl")
        xcel = devs[1].set_lang("aocl")
        tool = tool_table[name]
        super(VLAB, self).__init__(name, devs, host, xcel, tool)
    
    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote AWS machines)
    def compile(self, remote=False):
        pass

Platform.aws_f1 = AWS_F1()
Platform.zc706  = ZC706()
Platform.vlab   = VLAB()