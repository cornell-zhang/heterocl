import os
from . import devices

# Post process generated kernel and host 
def post_process(path):
    print(f"[  INFO  ] Post-processing generated code in {path}")
    # get post processed host and kernel code
    files = os.listdir(path)
    sources = [i for i in files if i.endswith('.cpp') or i.endswith('.cl')]
    host, kernel = None, None
    for f_name in sources:
        if "host" or "testbench" in f_name:
            host = os.path.join(path, f_name)
        if "kernel" in f_name:
            kernel = os.path.join(path, f_name)
    assert host is not None
    assert kernel is not None

    for k, v in devices.Project.post_proc_list.items():
        with open(host, "r+") as hfp, open(kernel, "r+") as kfp:
            # run post process functions
            print(f"  - {k}. process function {v}")
            host_code, kernel_code = v(path, hfp.read(), kfp.read())
            hfp.seek(0); hfp.truncate()
            kfp.seek(0); kfp.truncate()
            hfp.write(host_code)
            kfp.write(kernel_code)