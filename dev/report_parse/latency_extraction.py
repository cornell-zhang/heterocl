import os
import re
import json
import xmltodict

class report_dict:
  def __init__(self, dt):
    self._dict = dt

  def __setitem__(self, key, value):
    self._dict.__setitem__(key, value)

  def __getitem__(self, key):
    data = self._dict.__getitem__(key)
    return report_dict(data)

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

def parse_detail(path):
  xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
  if not os.path.isfile(xml_file):
    raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
  
  profile = json.loads(open(os.path.join(path,"report.json"), "r").read())
  clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
  
  summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]

  def get_keys(obj):
    first_stage = list(obj.keys())[0]
    word = re.split('\d+',first_stage)[0]
    keys = []
    for key in summary[first_stage].keys():
      if word not in key:
        keys.append(key)
    return keys

  keys = get_keys(summary)

  def loop(obj, key):
    init = {}
    for k, v in obj.items():
      val = v.get(key) + " " + clock_unit
      init[k] = { key.lower() : val }
      in_k, in_v = list(v.items())[-1]
      inner = {}
      idx = in_k
      first_in = True
      while isinstance(in_v, dict):
        te_v = in_v.get(key) + " " + clock_unit
        te = { in_k : { key.lower() : te_v } }
        if (first_in):
          inner = te
          first_in = False
        else:
          inner[idx].update(te)
          idx = in_k
        in_k, in_v = list(in_v.items())[-1]
      init[k].update(inner)    
    test = { key : init }
    return test

  res = {}
  for key in keys:
    res.update(loop(summary,key))
  out = report_dict(res)
  #print(out['Latency']['out_matrix_x'])
  #print(out['Latency']['out_matrix_x']['out_matrix_y'])
  #print(out['Latency']['out_matrix_x']['out_matrix_y']['out_matrix_k'])

parse_v2("./project/")
