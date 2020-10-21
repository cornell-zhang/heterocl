import os, re
import json
import time
import xmltodict
from tabulate import tabulate

class report_dict:
  def __init__(self, dt):
    self._dict = dt

  def __setitem__(self, key, value):
    self._dict.__setitem__(key, value)

  def __getitem__(self, key):
    data = self._dict.__getitem__(key)
    return report_dict(data)

  def __getattr__(self, attr):
    return self.__getitem__(attr)

  def __str__(self):
    target = list(self._dict.keys())[0]
    return f'{self.get(target)}'

  def update(self, *args, **kwargs):
    return self._dict.update(*args, **kwargs)

  def get(self, *args, **kwargs):
    return self._dict.get(*args, **kwargs)

  def keys(self):
    return self._dict.keys()

  def values(self):
    return self._dict.values()

  def items(self):
    return self._dict.items()

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

    # Latency information extraction
    clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
    summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]

    # Latency type
    def get_keys(obj):
      keys = []
      for v in obj.values():
        for cat in v.keys():
          if "_" not in cat and cat not in keys:
            keys.append(i)
      return keys

    # Latency value with respect to different format
    def get_value(self, key):
      num = self.get(key)
      if isinstance(num, str):
        val = str(num)
      elif isinstance(num, dict):
        val = str(num['range']['min']) + ' ~ ' + str(num['range']['max'])
      else:
        val = "N/A"
      return val

    # Extract out information
    def lat_info(obj, key):
      res = dict()
      for k, v in obj.items():
        val = get_value(v, key) 
        res[k] = { key.lower() : val }
        prev_k = k
        in_k, in_v = list(v.items())[-1]
        field = res
        while not isinstance(in_v, str):
          field = field[prev_k]
          val = get_value(in_v, key)
          entry = { in_k : { key.lower() : val } }
          field.update(entry)
          prev_k = in_k
          in_k, in_v = list(in_v.items())[-1]
      return res
     
    keys = get_keys(summary)
    fin = dict()
    for key in keys:
      res = lat_info(summary, key)
      fin.update({key : res})    
    final_dict = report_dict( fin )

    # Table information
    rows = []
    def tbl_info(obj, keys):
      for k, v in obj.items():
        row_entry = [ k ]
        for y in keys:
          row_entry.append(get_value(v, y))
        rows.append(row_entry)
        in_k, in_v = list(v.items())[-1]
        while not isinstance(in_v, str):
          if 'range' in list(in_v.keys()):
            break
          row_entry = [ in_k ]
          for y in keys:
            row_entry.append(get_value(in_v, y))
          rows.append(row_entry)
          in_k, in_v = list(in_v.items())[-1]

    # Table preparation
    headers = [ "Loop Name" ]
    lst = list(fin.items())
    for i in range(0,len(lst)):
      headers.append(lst[i][0])
    tbl_info(summary, keys)
    lat_tablestr = tabulate(rows, headers=headers, tablefmt="psql")

    if print_flag:
        print(table)
    return profile

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
    else:
        raise RuntimeError("tool {} not yet supported".format(target.tool.name))
