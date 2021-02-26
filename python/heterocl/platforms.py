import os, subprocess, json
from .devices import Platform, dev_table, tool_table
from os.path import expanduser

# Save information to HOME
def save_cache_info(fname, key, value):
    path = os.path.join(expanduser("~"), ".hcl")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, fname)
    with open(path) as fp:
        data = json.load(fp)
        data.update({key: value})
        json.dump(data, fp)

def get_cache_info(fname, key):
    path = os.path.join(expanduser("~"), ".hcl")
    path = os.path.join(path, fname)
    assert os.path.exists(path)
    with open(path) as fp:    
        data = json.load(fp)
        try:
            return data[key]
        except:
            return None

def clean_cache(fname):
    path = os.path.join(expanduser("~"), ".hcl")
    path = os.path.join(path, fname)
    os.remove(path)

def run_shell_script(command):
    with open("temp.sh", "w") as fp:
        fp.write(command)
    ret = subprocess.run(command, 
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
        instance_id = get_cache_info("aws.json", "instance_id")
        if instance_id is None:
            command = "aws ec2 run-instances --image-id {} \
                --security-group-ids sg-08111c9462c75f193 \
                --block-device-mapping DeviceName=/dev/sda1,Ebs={{VolumeSize=100}} \
                --instance-type {} --key-name {};".format(self.AMI_ID, instance, aws_key)
            out = run_shell_script(command)
            ret = json.loads(out)
            instance_id = ret["Instances"][0]["InstanceId"]
            save_cache_info("aws.json", "instance_id", instance_id)
        return instance_id
    
    def get_ip_addr(instance_id):
        command = "aws ec2 describe-instances --instance-ids {} \
            --query 'Reservations[0].Instances[0].PublicIpAddress' | sed 's/\"//g'".format(instance_id)
        ip_addr = run_shell_script(command)
        save_cache_info("aws.json", "ip_addr", ip_addr)
        return ip_addr

    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote AWS machines)
    def compile(self, remote=False, aws_key_path=None, instance="t2.micro"):
        # Generate project and utility files

        sys.exit()
        if remote:
            assert os.path.exists(aws_key_path)
            aws_key = aws_key_path.split("/")[-1].replace(".pem", "")

            # Create instances
            instance_id = self.create_instance(aws_key, instance)
            ip_addr = self.get_ip_addr(instance_id)

            # Upload to S3
            self.upload()

        else:
            # Compile locally
            self.tool.compile()
    
    def download(self):
        pass
    
    def delete_instance(self):
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