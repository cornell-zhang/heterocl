import os
import json
import xmltodict
from tabulate import tabulate

def parse_xml(path):
    xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
    assert os.path.isfile(xml_file), "Cannot find {}".format(xml_file)
    outfile = open(os.path.join(path,"profile.json"), "w")
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
        json.dump(profile, outfile, indent=2)
    res = {}
    res["HLS Version"] = "Vivado HLS " + profile["ReportVersion"]["Version"]
    res["Product family"] = profile["UserAssignments"]["ProductFamily"]
    res["Target device"] = profile["UserAssignments"]["Part"]
    clock_unit = profile["UserAssignments"]["unit"]
    res["Top Model Name"] = profile["UserAssignments"]["TopModelName"]
    res["Target Clock Period"] = profile["UserAssignments"]["TargetClockPeriod"] + clock_unit
    res["Latency"] = "Min {:>10} cycles\n".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Best-caseLatency"]) + \
                     "Max {:>10} cycles".format(
                         profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Worst-caseLatency"])
    est_resources = profile["AreaEstimates"]["Resources"]
    avail_resources = profile["AreaEstimates"]["AvailableResources"]
    resources = {}
    for name in ["BRAM_18K", "DSP48E", "FF", "LUT"]:
        item = [est_resources[name], avail_resources[name]]
        item.append("{}%".format(round(int(item[0])/int(item[1])*100)))
        resources[name] = item.copy()
    res["Resources"] = tabulate([[key] + resources[key] for key in resources.keys()],
                                headers=["Name", "Total", "Available", "Utilization"])
    lst = list(res.items())
    tablestr = tabulate(lst, tablefmt="psql").split("\n")
    endash = tablestr[0].split("+")
    splitline = "+" + endash[1] + "+" + endash[2] + "+"
    tablestr.insert(5, splitline)
    table = '\n'.join(tablestr)
    print(table)
    return profile