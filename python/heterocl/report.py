import os, re
import json
import time
import xmltodict
from tabulate import tabulate

# synthesis report in rpt format
def parse_vhls_report(path):
    assert os.path.exists(path), path
    res = dict()
    with open(path, "r") as fp:
        lines = fp.readlines()
        index = 0
        for line in lines:
            if "Clock" in line and "Target" in line:
                content = lines[index+2]
                clk_target, clk_estimate = [ 
                    float(_.replace("ns", "")) for _ in content.split("|")[2:4] ]
                res["clk_target"] = clk_target
                res["clk_estimate"] = clk_estimate

            if "Utilization Estimates" in line:
                content = lines[index+14]
                bram, dsp, ff, lut, uram = [
                    int(_) for _ in content.split("|")[2:7]
                ]

                # percentage data
                content = lines[index+22]
                bram_p, dsp_p, ff_p, lut_p, uram_p = [
                    float(_.replace("~", "")) for _ in content.split("|")[2:7]
                ]

                res["lut"] = [lut, lut_p]
                res["dsp"] = [dsp, dsp_p]
                res["ff"] = [ff, ff_p]
                res["bram"] = [bram, bram_p]
                res["uram"] = [uram, uram_p]

            if "+ Latency" in line:
                content = lines[index+6].split("|")
                latency_cyc = [ int(content[1]), int(content[2]) ]
                latency_abs = [ content[3], content[4] ]
                res["latency_cyc"] = latency_cyc
                res["latency_abs"] = latency_abs
            index += 1
    return res


# profiling result
def parse_vitis_prof_report(path):
    res = dict()
    prof = os.path.join(path, "../profile_summary.csv")
    assert os.path.exists(prof), prof

    with open(prof, "r") as fp:
        lines = fp.readlines()
        index = 0
        for line in lines:
            if "Top Kernel Execution" in line:
                res["runtime_ms"] = float(lines[index+2].split(",")[-4])
            if "Top Data Transfer" in line:
                content = lines[index+2].split(",")
                res["transfer_rate_mbps"] = content[-2]
                res["read_mb"] = content[-3]
                res["write_mb"] = content[-4]
                res["tranfer_efficiency"] = content[-6]
                res["byte_per_transfer"] = content[-7]
                res["transfer_num"] = content[-8]
            index += 1
    
    info = os.path.join(path, "kernel.xclbin.info")
    assert os.path.exists(info), info
    with open(info, "r") as fp:
        lines = fp.readlines()
        index = 0
        for line in lines:
            if "DATA_CLK" in line:
                freq = lines[index+3].replace("Frequency:", "")
                freq = freq.replace("MHz", "")
                res["freq"] = float(freq)
            index += 1
        
    return res


def parse_js(path, print_flag=False):
    js_file = os.path.join(path, "kernel/reports/lib/report_data.js")
    if not os.path.isfile(js_file):
        raise RuntimeError("Cannot find {}, run csyn first".format(js_file))

    # TODO: parse AOCL profiling report 
    with open(js_file, "r") as fp:
        js_scripts = fp.read()
        regex = "total_kernel_resources.*?(\d+), (\d+), (\d+), (\d+), (\d+)"
        match = re.findall(regex, js_scripts)
        print("[{}] Parsing AOCL HLS report... ".format(
            time.strftime("%H:%M:%S", time.gmtime())))
        LUT, FF, RAM, DSP, MLAB = match[0]
        print("[--------] ALUT : {}".format(LUT))
        print("[--------] FF   : {}".format(FF))
        print("[--------] RAM  : {}".format(RAM))
        print("[--------] DSP  : {}".format(DSP))
        print("[--------] MLAB : {}".format(MLAB))
    

def parse_xml(path, print_flag=False):
    xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
    if not os.path.isfile(xml_file):
        raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
    json_file = os.path.join(path,"report.json")
    outfile = open(json_file, "w")
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
        json.dump(profile, outfile, indent=2)
    res = {}
    res["HLS Version"] = "Vivado HLS " + profile["ReportVersion"]["Version"]
    res["Product family"] = profile["UserAssignments"]["ProductFamily"]
    res["Target device"] = profile["UserAssignments"]["Part"]
    clock_unit = profile["UserAssignments"]["unit"]
    res["Top Model Name"] = profile["UserAssignments"]["TopModelName"]
    res["Target CP"] = profile["UserAssignments"]["TargetClockPeriod"] + " " + clock_unit
    res["Estimated CP"] = profile["PerformanceEstimates"]["SummaryOfTimingAnalysis"]["EstimatedClockPeriod"] + " " + clock_unit
    res["Latency (cycles)"] = "Min {:<6}; ".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Best-caseLatency"]) + \
                              "Max {:<6}".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Worst-caseLatency"])
    res["Interval (cycles)"] = "Min {:<6}; ".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Interval-min"]) + \
                               "Max {:<6}".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Interval-max"])
    est_resources = profile["AreaEstimates"]["Resources"]
    avail_resources = profile["AreaEstimates"]["AvailableResources"]
    resources = {}
    for name in ["BRAM_18K", "DSP48E", "FF", "LUT"]:
        item = [est_resources[name], avail_resources[name]]
        item.append("{}%".format(round(int(item[0])/int(item[1])*100)))
        resources[name] = item.copy()
    res["Resources"] = tabulate([[key] + resources[key] for key in resources.keys()],
                                headers=["Type", "Used", "Total", "Util"],
                                colalign=("left","right","right","right"))
    lst = list(res.items())
    tablestr = tabulate(lst, tablefmt="psql").split("\n")
    endash = tablestr[0].split("+")
    splitline = "+" + endash[1] + "+" + endash[2] + "+"
    tablestr.insert(5, splitline)
    table = '\n'.join(tablestr)
    if print_flag:
        print(table)
    return profile


# Entry function to get performance number
def report_stats(target, folder):
    path = folder
    if target.tool.name == "vivado_hls":
        if os.path.isdir(os.path.join(path, "out.prj")):
            return parse_xml(path)
        else:
            raise RuntimeError("Not found out.prj folder")

    elif target.tool.name == "aocl":
        if os.path.isdir(os.path.join(path, "kernel/reports")):
            return parse_js(path)

    elif target.tool.name == "vitis":
        path = target.tool.hls_report_dir
        return path

    else:
        raise RuntimeError("tool {} not yet supported".format(target.tool.name))
