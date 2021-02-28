import os, subprocess, json, time
from .devices import Platform, CPU, FPGA, PIM, Project
from .tools import *
from os.path import expanduser


# Save information to HOME
def save_cache_info(fname, key, value):
    path = os.path.join(expanduser("~"), ".hcl")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, fname)
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            json.dump({}, f)
    with open(path, 'r+') as fp:
        data = json.load(fp)
        fp.seek(0)
        fp.truncate()
        data.update({key: value})
        json.dump(data, fp)


def get_cache_info(fname, key):
    path = os.path.join(expanduser("~"), ".hcl")
    path = os.path.join(path, fname)
    try:
        assert os.path.exists(path)
        with open(path) as fp:    
            data = json.load(fp)
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


def run_shell_remote(command, inputs, ssh):
    command = command.format(**inputs)
    with open("run.sh", "w") as fp:
        fp.write(command)
    command = "{} 'bash -s' < run.sh".format(ssh)
    ret = subprocess.run(command, 
        stdout=subprocess.PIPE, check=True, shell=True)
    os.remove("run.sh")
    return ret.stdout.decode('utf-8')


# define some built-in platforms in HCL
class AWS_F1(Platform):
    def __init__(self):
        name = "aws_f1"
        devs = [
            CPU("intel", "e5"), 
            FPGA("xilinx", "xcvu19p")
            ]
        host = devs[0].set_lang("xocl")
        xcel = devs[1].set_lang("vhls")
        tool = Tool.vitis

        self.AMI_ID = "ami-0a7b98fdb062be15f"
        self.XPFM = "xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm"
        super(AWS_F1, self).__init__(name, devs, host, xcel, tool)

        self.cache = None
        self.tool = tool

    # copt tool specific utility files
    def copy_utility(self, path, source):
        self.tool.copy_utility(path, source)

    # upload the work project to AWS
    def upload(self, work_path, project_name):
        base = "test-hcl-" + os.getlogin()
        command = "aws s3 rm s3://{base}/{project_name}; \
            aws s3 mb s3://{base}/{project_name}; \
            aws s3 cp --recursive {work_path} s3://{base}/{project_name}/".\
                format(project_name=project_name, 
                    work_path=work_path, base=base)
        save_cache_info(self.cache, "s3", project_name)
        run_shell_script(command)

    def create_instance(self, aws_key, instance):
        instance_id = get_cache_info(self.cache, "instance_id")
        if instance_id is None:
            command = "aws ec2 run-instances --image-id {} \
                --security-group-ids sg-08111c9462c75f193 \
                --block-device-mapping DeviceName=/dev/sda1,Ebs={{VolumeSize=100}} \
                --instance-type {} --key-name {};".format(self.AMI_ID, instance, aws_key)
            out = run_shell_script(command)
            ret = json.loads(out)
            instance_id = ret["Instances"][0]["InstanceId"]
            save_cache_info(self.cache, "instance_id", instance_id)
        return instance_id
    
    def get_ip_addr(self, instance_id):
        command = "aws ec2 describe-instances --instance-ids {} \
            --query 'Reservations[0].Instances[0].PublicIpAddress' | sed 's/\"//g'".format(instance_id)
        ip_addr = run_shell_script(command).replace('\n','')
        save_cache_info(self.cache, "ip_addr", ip_addr)
        return ip_addr
    
    # Pull back aws-fpga and run syn
    def remote_compile(self, ip_addr, project_name):
        key_path = get_cache_info(self.cache, "key_path")
        base = "test-hcl-" + os.getlogin()

        # Get compilation mode
        mode = self.tool.mode
        if mode == "hw_exe": tool_mode = "hw"
        elif mode == "sw_sim": tool_mode = "sw_emu"
        elif mode == "hw_sim": tool_mode = "hw_emu"

        inputs = {
            "key_path": key_path,
            "ip_addr" : ip_addr,
            "project_name": project_name,
            "base": base,
            "tool_mode": tool_mode
        }
        ssh = "ssh -i {key_path} centos@{ip_addr} ".format(**inputs)
        curr_time = time.strftime("%H:%M:%S", time.gmtime())
        print("[{}] Log in with {}".format(curr_time, ssh))
        print("[{}] Mode ({}). Project name ({})".format(mode, project_name))

        command = "ssh -i {key_path} -o \"StrictHostKeyChecking no\" \
            centos@{ip_addr} uptime".format(**inputs)
        run_shell_script(command)
        command = "scp -i {key_path} -r ~/.aws centos@{ip_addr}:~".format(**inputs)
        run_shell_script(command)
                
        # clone aws-fpga to HOME
        command = """
cd ~; 
if [ ! -d "aws-fpga" ]; then
    git clone https://github.com/aws/aws-fpga.git;
    echo "source $HOME/aws-fpga/vitis_setup.sh" >> ~/.bashrc
    echo "source $HOME/aws-fpga/vitis_runtime_setup.sh" >> ~/.bashrc
fi
"""
        print("[ INFO ] Initializing AWS environment...")
        run_shell_remote(command, inputs, ssh)

        command = """
cd ~; 
if [ ! -d "{project_name}" ]; then
    mkdir {project_name}; 
    aws s3 cp --recursive s3://{base}/{project_name} {project_name}; 
fi
cd {project_name}; 
make all TARGET={tool_mode} DEVICE=$AWS_PLATFORM
"""
        run_shell_remote(command, inputs, ssh)
        save_cache_info(self.cache, "status", "finished")

        # Register the AFI image
        command = """
cd ~/{project_name};
cp build_dir.hw.*/kernel.xclbin .
if [ ! -f *afi_id.txt ]; then
    aws s3 rm s3://{base}/{project_name}/afi
    aws s3 mb s3://{base}/{project_name}/afi
    $VITIS_DIR/tools/create_vitis_afi.sh -xclbin=kernel.xclbin \
        -s3_bucket=hcl-test-afi -s3_dcp_key=$TARGET
    aws s3 cp host s3://{base}/{project_name}/
    aws s3 cp kernel.awsxclbin s3://{base}/{project_name}/
fi
AFI_ID=$(cat *afi_id.txt | grep -Eo '(afi-.+)' | sed 's/",//g')
aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID
echo $AFI_ID
"""
        out = run_shell_remote(command, inputs, ssh)
        print(out)
        afi_id = out.split("\n")[-1]
        save_cache_info(self.cache, "status", "registering")
        save_cache_info(self.cache, "afi_id", afi_id)

    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote AWS machines)
    def compile(self, args, remote=False, 
            aws_key_path=None, instance="t2.micro", xpfm=None):

        work_path = Project.path
        project_name = Project.project_name
        self.cache = "{}-aws.json".format(project_name)
        if remote:
            assert os.path.exists(aws_key_path)
            save_cache_info(self.cache, "key_path", aws_key_path)
            aws_key = aws_key_path.split("/")[-1].replace(".pem", "")

            # Create instances
            instance_id = self.create_instance(aws_key, instance)
            ip_addr = self.get_ip_addr(instance_id)

            # Upload to S3 bucket
            status = get_cache_info(self.cache, "status")
            if status is None:
                self.upload(work_path, project_name)
                print("[ INFO ] Uploaded source code to AWS S3. Start compiling")
                # Do we want to make this compilation running background?
                self.remote_compile(ip_addr, project_name)
                
            else:
                print("[ INFO ] Compilation running on AWS. Check status with \
                    ssh -i {} centos@{}...".format(aws_key_path, ip_addr))

        else:
            # Compile locally using the same tool chain
            if xpfm is not None:
                self.XPFM = xpfm
            self.tool.compile(work_path, self.XPFM)
    
    def wait_afi_register(self):
        afi_id = get_cache_info(self.cache, "afi_id")
        command = "aws ec2 describe-fpga-images --fpga-image-ids {}".format(afi_id)
        out = run_shell_script(command)
        print(out); sys.exit()

    # Execute the program and fetch JSON
    def execute(self, work_path, mode, remote=False, 
            aws_key_path=None, instance="f1.2xlarge"):

        if remote:
            aws_key_path = get_cache_info(self.cache, "key_path")
            assert aws_key_path is not None
            aws_key = aws_key_path.split("/")[-1].replace(".pem", "")

            # Run the host program
            mode = self.tool.mode         
            if mode == "hw_exe":
                run_cmd = "./host kernel.xclbin"
            elif mode == "sw_sim":
                run_cmd = "XCL_EMULATION_MODE=sw_emu ./host kernel.xclbin"
            elif mode == "hw_sim":
                run_cmd = "XCL_EMULATION_MODE=hw_emu ./host kernel.xclbin"

            # Check if the AFI image is registered
            if mode == "hw_exe":
                while self.wait_afi_register():
                    sleep(10 * 60)
                
                # Create F1 instance and start running  
        else:
            pass

    
    def delete_instance(self):
        pass

class ZC706(Platform):
    def __init__(self):
        name = "zc706"
        devs = [
            CPU("arm", "a9"), 
            FPGA("xilinx", "xc7z045")
        ]
        host = devs[0].set_lang("vhls")
        xcel = devs[1].set_lang("vhls")
        tool = Tool.vivado_hls
        super(ZC706, self).__init__(name, devs, host, xcel, tool)

class VLAB(Platform):
    def __init__(self):
        name = "vlab"
        devs = [
            CPU("intel", "e5"), 
            FPGA("intel", "arria10")
            ]
        host = devs[0].set_lang("aocl")
        xcel = devs[1].set_lang("aocl")
        tool = Tool.aocl
        super(VLAB, self).__init__(name, devs, host, xcel, tool)
    
    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote AWS machines)
    def compile(self, remote=False):
        pass

class U280(Platform):
    def __init__(self):
        name = "u280"
        devs = [
            CPU("intel", "e5"), 
            FPGA("xilinx", "xcvu19p")
            ]
        host = devs[0].set_lang("xocl")
        xcel = devs[1].set_lang("vhls")
        tool = Tool.vitis

        self.XFPM = "xilinx_u280_xdma_201920_3.xpfm"
        super(U280, self).__init__(name, devs, host, xcel, tool)

        self.cache = None
        self.tool = tool

    # Only support compiling locally
    def compile(self, args, xpfm=None):
        if xpfm is not None:
            self.XFPM = xpfm
        work_path = Project.path
        mode = self.tool.mode
        self.tool.compile(work_path, mode, self.XFPM)
    
    def copy_utility(self, path, source):
        self.tool.copy_utility(path, source)
        
    def execute(self, work_path, mode, **kwargs):
        self.tool.execute(work_path, mode)

Platform.aws_f1 = AWS_F1()
Platform.zc706  = ZC706()
Platform.vlab   = VLAB()
Platform.u280   = U280()