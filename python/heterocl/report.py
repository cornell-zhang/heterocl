import os
import json
import xmltodict
from tabulate import tabulate

def parse_xml(path,print_flag=False):
    xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
    if not os.path.isfile(xml_file):
        raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
    json_file = os.path.join(path,"profile.json")
    if os.path.isfile(json_file):
        profile = json.loads(open(json_file,"r").read())
    else:
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
    res["Latency"] = "Min {:>5} cycles\n".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Best-caseLatency"]) + \
                     "Max {:>5} cycles".format(
                         profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Worst-caseLatency"])
    res["Interval"] = "Min {:>5} cycles\n".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Interval-min"]) + \
                     "Max {:>5} cycles".format(
                         profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Interval-max"])
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