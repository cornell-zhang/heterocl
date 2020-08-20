import os
import json
import xmltodict

#https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
#https://blog.anvetsu.com/posts/custom-dictionary-type-python/
class report_dict(dict):
  def __init__(self):
    self.first_idx = ''
    #self.target_idx = True

  def __setitem__(self, key, value):
    super().__setitem__(key, value)

  def __getitem__(self, key):
    #obj = self.__dict__[key]
    #obj = super().__getitem__(key)
    #return obj.keys()
    #return obj[list(obj.keys())[0]]
    
    #if (self.target_idx):
    #  self.set_basekey(key)

    x = self.get(key)
    #print(x.keys())

    if (len(x) > 2):
      return x
    elif (self.first_idx.lower() in x.keys()):
      return self.get(self.first_idx.lower())
    else:
      obj = super().__getitem__(key)
      return obj[list(obj.keys())[0]]

  def update(self, *args, **kwargs):
    return super().update(*args, **kwargs)

  def get(self, *args, **kwargs):
    return super().get(*args, **kwargs)

  def keys(self):
    return super().keys()

  def values(self):
    return super().values()

  def items(self):
    return super().items()

  #https://stackoverflow.com/questions/23944657/typeerror-method-takes-1-positional-argument-but-2-were-given/23944658
  def set_basekey(self, key):
    self.first_idx = key
    #self.target_idx = False

def parse_v2(path):
  xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
  if not os.path.isfile(xml_file):
    raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
  
  profile = json.loads(open(os.path.join(path,"report.json"), "r").read())
  clock_unit = profile["UserAssignments"]["unit"]

  #out = {}
  out = report_dict()
  
  summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
  keys = ['TripCount', 'Latency', 'IterationLatency']

  def loop(obj, key):
    #test = {}
    test = report_dict()
    for k, v in obj.items():
      val = v.get(key) + " " + clock_unit
      init = { key.lower() : val }
      test[k] = init

      for e, i in v.items():
        if isinstance(i, dict):
          inner_val = i.get(key) + " " + clock_unit
          inner = { e : { key.lower() : inner_val } }
          init.update(inner) # change by custom_dict
    return test

  for key in keys:
    out[key] = loop(summary,key)
  
  out.set_basekey(keys)
  print(out.first_idx)

  #print(out) # checked correct

  #print('/\/\/\/\/\/\/\/\/\/\/\/\/')
  #print('========[Latency]========')
  #print(out['Latency'])
  #print('==== +[out_matrix_x]=====')
  #print(out['Latency']['out_matrix_x'])
  #print('==== +[out_matrix_y]=====')
  #print(out['Latency']['out_matrix_x']['out_matrix_y'])

  #print(out['Latency']['out_matrix_x'].keys())
  #interest = out['Latency']['out_matrix_x']
  #print(interest[list(interest.keys())[0]])

  #finalfile = open('final.json', 'w')
  #json.dump(out, finalfile, indent=2)

parse_v2("./project/")
