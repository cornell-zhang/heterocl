import os
import json
import xmltodict

def parse(path):
  xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
  if not os.path.isfile(xml_file):
    raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
  
  profile = json.loads(open(os.path.join(path,"report.json"), "r").read())
  clock_unit = profile["UserAssignments"]["unit"]

  def extract_values(obj, key, target_key):
    arr = {}
    def extract(obj, arr, key):
      for k, v in obj.items():
        if (target_key in k):
          arr[k] = v[key] + " " + clock_unit
        if isinstance(v, dict):
          extract(v, arr, key)
      return arr
    return extract(obj, arr, key)

  return extract_values(profile["PerformanceEstimates"]["SummaryOfLoopLatency"], "Latency", "X")

r = parse("./project/")
print(r)
print(r[list(r.keys())[0]])
