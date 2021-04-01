import os, subprocess, json, time, sys, re
from .devices import Platform, CPU, FPGA, ASIC, PIM, Project
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
    # os.remove("run.sh")
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
            time.sleep(10)
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
        save_cache_info(self.cache, "ssh", ssh)

        curr_time = time.strftime("%H:%M:%S", time.gmtime())
        print("[{}] Log in with {}".format(curr_time, ssh))
        print("[{}] Mode ({}). Project name ({})".format(curr_time, mode, project_name))

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
fi
"""
        print("[ INFO ] Initializing AWS environment...")
        run_shell_remote(command, inputs, ssh)

        # Start the compilation (in the background)
        command = """
source $HOME/aws-fpga/vitis_setup.sh
source $HOME/aws-fpga/vitis_runtime_setup.sh
cd ~; 
if [ ! -d "{project_name}" ]; then
    mkdir {project_name}; 
    aws s3 cp --recursive s3://{base}/{project_name} {project_name}; 
fi
cd {project_name}; 
make all TARGET={tool_mode} DEVICE=$AWS_PLATFORM &> output.log &
echo "$!" > PID.txt
"""
        run_shell_remote(command, inputs, ssh)
        save_cache_info(self.cache, "status", "running")


    def check_process_liveness(self, ip_addr, project_name):
        ssh = get_cache_info(self.cache, "ssh")
        inputs = {"project_name": project_name}
        command = """
cd ~/{project_name}; 
PID=$(cat PID.txt)
if [ -n "$PID" -a -e /proc/$PID ]; then
    echo "process exists"
    ps -p $PID -o etime    
fi
"""     
        out = run_shell_remote(command, inputs, ssh)
        is_live = False
        run_time = 0
        if "process exists" in out:
            is_live = True
            run_time = out.split("ELAPSED")[-1].replace("\n", "").lstrip()
        else:
            mode = self.tool.mode
            if mode == "hw_exe": tool_mode = "hw"
            elif mode == "sw_sim": tool_mode = "sw_emu"
            elif mode == "hw_sim": tool_mode = "hw_emu"
            command = """
cd ~/{project_name};
cp build_dir.{tool_mode}*/kernel.xclbin .
"""
            inputs = {"project_name": project_name, "tool_mode": tool_mode}
            run_shell_remote(command, inputs, ssh)

        return is_live, run_time

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

            # Create instances (return IP addr if it is running)
            instance_id = self.create_instance(aws_key, instance)
            ip_addr = self.get_ip_addr(instance_id)

            # Upload to S3 bucket if instance does not exists
            status = get_cache_info(self.cache, "status")
            if status is None:
                self.upload(work_path, project_name)
                print("[  INFO  ] Uploaded source code to AWS S3. Start compiling...")
                self.remote_compile(ip_addr, project_name)
                sys.exit()

            elif status == "running":
                ssh = "ssh -i {} centos@{}".format(aws_key_path, ip_addr)
                # Check process runtime (PID saved in project folder)
                is_live, runtime = self.check_process_liveness(ip_addr, project_name)

                if is_live:
                    print("[  INFO  ] Compiling (Elaspsed {})".format(runtime))
                    print("[  INFO  ] SSH {}".format(ssh))
                    sys.exit()
                
                else:
                    mode = self.tool.mode
                    if mode == "hw_exe": 
                        # Start AFI registeration
                        self.register_afi(ip_addr, project_name)
                        # Waiting for AFI registration
                        self.wait_afi_register()
                    else:
                        print("[  INFO  ] Compilation done. Start Emulation")
                        save_cache_info(self.cache, "status", "done")

            else:
                print("[  INFO  ] Compilation running on AWS. Check status with \
                    ssh -i {} centos@{}...".format(aws_key_path, ip_addr))

        else:
            # Compile locally using the same tool chain
            if xpfm is not None:
                self.XPFM = xpfm
            self.tool.compile(work_path, self.XPFM)
    
    def register_afi(ip_addr, project_name):
        ssh = get_cache_info(self.cache, "ssh")
        base = "test-hcl-" + os.getlogin()
        inputs = {
            "base": base,
            "project_name": project_name
        }
        # Register the AFI image
        command = """
source $HOME/aws-fpga/vitis_setup.sh
source $HOME/aws-fpga/vitis_runtime_setup.sh
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
echo $AFI_ID > AFI_ID.txt
"""
        out = run_shell_remote(command, inputs, ssh)
        afi_id = out.split("\n")[-1]
        print("[  INFO  ] AFI ID registered {}".format(afi_id))
        save_cache_info(self.cache, "status", "registering")
        save_cache_info(self.cache, "afi_id", afi_id)

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

            status = get_cache_info(self.cache, "status")
            if status == "done":
                # Run the host program
                mode = self.tool.mode   
                project_name = Project.project_name
                run_cmd = """
source $HOME/aws-fpga/vitis_setup.sh
source $HOME/aws-fpga/vitis_runtime_setup.sh
cd {project_name}
{execute_cmd}
"""    
                ssh = get_cache_info(self.cache, "ssh")
                if mode == "sw_sim":
                    execute_cmd = "XCL_EMULATION_MODE=sw_emu ./host kernel.xclbin"
                    inputs = {"project_name": project_name, "execute_cmd": execute_cmd}
                    out = run_shell_remote(run_cmd, inputs, ssh)
                    self.copy_data_back(project_name)

                elif mode == "hw_sim":
                    execute_cmd = "XCL_EMULATION_MODE=hw_emu ./host kernel.xclbin"
                    inputs = {"project_name": project_name, "execute_cmd": execute_cmd}
                    out = run_shell_remote(run_cmd, inputs, ssh)
                    self.copy_data_back(project_name)

                # Check if the AFI image is registered
                elif mode == "hw_exe":
                    while self.wait_afi_register():
                        sleep(10 * 60)

                    # Create F1 instance and start running  
            else:
                print("[  INFO  ] Instance not done yet. Status {}".format(status))
        else:
            pass

    def copy_data_back(self, project_name):
        print("[  INFO  ] Copying result back...")
        aws_key_path = get_cache_info(self.cache, "key_path")
        ip_addr = get_cache_info(self.cache, "ip_addr")
        assert aws_key_path is not None
        command = "scp -i {aws_key_path} centos@{ip_addr}:~/{project_name}/inputs.json ."
        run_shell_script(command)
    
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
        
    def copy_utility(self, path, source):
        self.tool.copy_utility(path, source)
    
    def compile(self, args, **kwargs):
        print("[  INFO  ] Checking g++ tool version for VHLS csim")
        assert os.system("which vivado_hls >> /dev/null") == 0, \
            "Please source Vivado HLS setup scripts before running"
        out = run_shell_script("g++ --version")
        pattern = "\d\.\d\.\d"
        ver = re.findall(pattern, out)[0].split(".")
        assert int(ver[0]) * 10 + int(ver[1]) >= 48, \
            "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

    def execute(self, work_path, mode):
        return self.tool.execute(work_path, mode)

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
    
    def remote_compile(self, project_name, instance):
        ssh = get_cache_info(self.cache, "ssh")
    
        mode = self.tool.mode
        compile_mode = ""
        if mode == "sw_sim":
            compile_mpde = "-march=emulator"
        elif mode == "hw_sim":
            raise RuntimeError("-march=simulator only supported for AOC > 19")

        if instance == "fpga-pac-a10":
            target = "pac_a10"
            AOC = "/export/fpga/tools/quartus_pro/17.1.1/hld"

        elif instance == "fpga-pac-s10":
            target = "pac_s10_dc"
            AOC = "/export/fpga/tools/quartus_pro/18.1.2_patches_202_203_206/hld" 
        
        elif instance == "fpga-bdx-opencl":
            target = "bdw_fpga_v1.0"
            AOC = "/export/fpga/tools/quartus_pro/16.0.2/hld/"

        inputs = {
            "project_name": project_name,
            "target": target,
            "AOC": AOC,
            "compile_mode": compile_mode,
        }

        aoc_command = """
cd $HOME/{project_name}
aoc -board={target} -time time.out \
    -time-passes {compile_mode} \
    -regtest_mode -v -fpc \
    -fp-relaxed -opt-arg -nocaching -report \
    -profile=all -I {AOC}/include/kernel_headers \
    kernel.cl
" >> qsub-run.sh
""".format(**inputs)

        # Patches for old version AOC
        if instance == "fpga-bdx-opencl":
            aoc_command = """
cd $HOME/{project_name}
aoc --board {target} --time time.out \
    --time-passes {compile_mode} \
    --regtest_mode -v --fpc \
    --fp-relaxed --opt-arg --nocaching --report \
    --profile all -I {AOC}/include/kernel_headers \
    kernel.cl
" >> qsub-run.sh
""".format(**inputs)            

        inputs = {
            "instance": instance,
            "project_name": project_name,
            "aoc_command": aoc_command
        }

        command = """
cd $HOME/{project_name}
if [ -f qsub-run.sh ]; then
    rm qsub-run.sh
fi

echo "#!/bin/bash
#PBS -l select=1:ncpus=24 -q xeon
source /export/fpga/bin/setup-fpga-env {instance}
{aoc_command}

qsub -v "path=$PWD" -l walltime=36:00:00 -N {project_name} -o aocl.out \
    -e aocl.err qsub-run.sh
"""
        out = run_shell_remote(command, inputs, ssh)
        save_cache_info(self.cache, "status", "running")
    
    def upload(self, work_path, project_name):
        key_path = get_cache_info(self.cache, "key_path")
        ssh = get_cache_info(self.cache, "ssh")
        command = """
cd $HOME
if [ ! -d {project_name} ]; then
    mkdir {project_name}
fi
"""
        inputs = {"project_name": project_name}
        run_shell_remote(command, inputs, ssh)
        command = "scp -i {} -r {}/* sx233@ssh-iam.intel-research.net:~/{}/".\
            format(key_path, project_name, project_name)
        run_shell_script(command)
    

    def check_process_liveness(self, project_name):
        command = "/opt/pbs/default/bin/qstat -u sx233"
        ssh = get_cache_info(self.cache, "ssh")
        out = run_shell_remote(command, {}, ssh)
        if project_name[:10] in out:
            print("[  INFO  ] Compilation running...")
            print(out)
            return True, 0
        else:
            return False, 0

    # Used for bitstream compiling (if remote is true, 
    # then we compile on remote VLAB machines)
    def compile(self, args, remote=False, 
            ssh_key_path=None, instance="fpga-pac-s10"):

        work_path = Project.path
        project_name = Project.project_name
        self.cache = "{}-vlab.json".format(project_name)

        if remote:
            assert os.path.exists(ssh_key_path)
            save_cache_info(self.cache, "key_path", ssh_key_path)
            ssh = "ssh -i {} sx233@ssh-iam.intel-research.net".format(ssh_key_path)
            save_cache_info(self.cache, "ssh", ssh)

            # Upload to S3 bucket if instance does not exists
            status = get_cache_info(self.cache, "status")
            if status is None:
                self.upload(work_path, project_name)
                print("[  INFO  ] Uploaded source code to VLAB. Start compiling...")
                self.remote_compile(project_name, instance)

            elif status == "running":
                is_live, runtime = self.check_process_liveness(project_name)
                if is_live:
                    print("[  INFO  ] SSH {}".format(ssh))
                    self.check_hls_report(project_name)
                else:
                    save_cache_info(self.cache, "status", "done")

        else:
            # Compile locally using the same tool chain
            if xpfm is not None:
                self.XPFM = xpfm
            self.tool.compile(work_path, self.XPFM)
            
    def copy_utility(self, path, source):
        self.tool.copy_utility(path, source)

    def execute(self, work_path, mode, remote=False, 
            ssh_key_path=None, instance="fpga-pac-s10"):

        if remote:
            status = get_cache_info(self.cache, "status")
            if status == "done":
                # Run the host program
                mode = self.tool.mode   
                project_name = Project.project_name
                run_cmd = """
cd $HOME/{project_name}
if [ -f qsub-execute.sh ]; then
    rm qsub-execute.sh
fi

echo "#!/bin/bash
#PBS -l select=1:ncpus=12 -q {instance}
source /export/fpga/bin/setup-fpga-env {instance}

cd $HOME/{project_name}
make host
aocl program acl0 kernel.aocx
{execute_cmd}
" >> qsub-execute.sh

qsub -v "path=$PWD" -l walltime=36:00:00 -N {project_name} -o fpga.out \
    -e fpga.err qsub-execute.sh
"""    
                ret = True
                ssh = get_cache_info(self.cache, "ssh")
                if mode == "sw_sim":
                    execute_cmd = "env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./host"
                    inputs = {"project_name": project_name, 
                              "execute_cmd": execute_cmd,
                              "instance": instance}
                    out = run_shell_remote(run_cmd, inputs, ssh)
                    self.copy_data_back(project_name)

                elif mode == "hw_sim":
                    execute_cmd = "env CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./host"
                    inputs = {"project_name": project_name, 
                              "execute_cmd": execute_cmd,
                              "instance": instance}
                    out = run_shell_remote(run_cmd, inputs, ssh)
                    self.copy_data_back(project_name)

                # Check if the AFI image is registered
                elif mode == "hw_exe":
                    execute_cmd = "./host kernel.aocx"
                    inputs = {"project_name": project_name, 
                              "execute_cmd": execute_cmd,
                              "instance": instance}
                    out = run_shell_remote(run_cmd, inputs, ssh)
                    print(out)
                    self.copy_data_back(project_name) 
                return ret
            else:
                print("[  INFO  ] Instance not done yet. Status {}".format(status))
                ret = False; return ret
        else:
            self.tool.execute(work_path, mode)
            ret = True; return ret
    
    # Download HLS report back if exists
    def check_hls_report(self, project_name):
        ssh = get_cache_info(self.cache, "ssh")
        command = """
cd $HOME/{project_name}
if [ -d kernel/reports ]; then 
    echo "Found HLS reports"
fi
"""
        inputs = {"project_name": project_name}
        out = run_shell_remote(command, inputs, ssh)
        if "Found HLS reports" in out:
            if not os.path.isdir("{}/reports".format(project_name)):
                key_path = get_cache_info(self.cache, "key_path")
                command = "scp -i {} -r sx233@ssh-iam.intel-research.net:~/{}/kernel/reports {}/".\
                    format(key_path, project_name, project_name)
                run_shell_script(command)
                print("[  INFO  ] Found HLS report. Downloading...")
            print("[  INFO  ] HLS report path {}/reports".format(project_name))
    
    def copy_data_back(self, project_name):
        print("[  INFO  ] Copying result back...")
        key_path = get_cache_info(self.cache, "key_path")
        command = "scp -i {} sx233@ssh-iam.intel-research.net:~/{}/*.json {}/".\
            format(key_path, project_name, project_name)
        run_shell_script(command)
        command = "scp -i {} sx233@ssh-iam.intel-research.net:~/{}/*.out {}/".\
            format(key_path, project_name, project_name)
        run_shell_script(command)
        command = "scp -i {} sx233@ssh-iam.intel-research.net:~/{}/*.err {}/".\
            format(key_path, project_name, project_name)
        run_shell_script(command)

        # Also copy the report back if MODE is hw_exe
        if self.tool.mode == "hw_exe":
            command = "scp -i {} \
                sx233@ssh-iam.intel-research.net:~/{}/kernel/acl_quartus_report.txt {}/".\
                format(key_path, project_name, project_name)
            run_shell_script(command)            
    
    def report(self, project_name):
        self.tool.report(project_name)

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
        return self.tool.execute(work_path, mode)

# Mainly used for AutoSA compilation
class Docker(U280):
    def __init__(self):
        super(Docker, self).__init__()

class Insider(AWS_F1):
    def __init__(self):
        super(Insider, self).__init__()
        self.AMI_ID = "ami-04c8ef1dee652fe24"

class U250(U280):
    def __init__(self):
        super(U250, self).__init__()
        self.XFPM = "xilinx_u250_qdma_201920_1.xpfm"   

class ASIC_HLS(Platform):
    def __init__(self):
        name = "asic_hls"
        devs = [
            ASIC("mentor", "catapultc"), 
            ASIC("mentor", "catapultc")
            ]
        host = devs[0].set_lang("catapultc")
        xcel = devs[1].set_lang("catapultc")
        tool = Tool.catapultc
        super(ASIC_HLS, self).__init__(name, devs, host, xcel, tool)
    
    def copy_utility(self, path, source):
        self.tool.copy_utility(path, source)
    
    def execute(self, work_path, mode):
        return self.tool.execute(work_path, mode)


Platform.aws_f1  = AWS_F1()
Platform.zc706   = ZC706()
Platform.vlab    = VLAB()
Platform.u280    = U280()
Platform.u250    = U250()
Platform.insider = Insider()
Platform.docker  = Docker()
Platform.asic_hls = ASIC_HLS()