import os
import json
import xmltodict

r = {}

def parse(path):
  xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
  if not os.path.isfile(xml_file):
    raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
  
  profile = json.loads(open(os.path.join(path,"report.json"), "r").read())
  clock_unit = profile["UserAssignments"]["unit"]

  out = {}

  def extract_values(obj, key):
    arr = {}
    def extract(obj, arr, key):
      for k, v in obj.items():
        if isinstance(v, dict):
          arr[k] = v[key] + " " + clock_unit
          extract(v, arr, key)
      return arr
    return extract(obj, arr, key)

  #out["TripCount"] = extract_values(profile["PerformanceEstimates"]["SummaryOfLoopLatency"], "TripCount")
  out["Latency"] = extract_values(profile["PerformanceEstimates"]["SummaryOfLoopLatency"], "Latency")
  #out["IterationLatency"] = extract_values(profile["PerformanceEstimates"]["SummaryOfLoopLatency"], "IterationLatency")
  return out

#r = parse("./project/")
#print(r)

#savefile = open("save.json", "w")
#json.dump(r, savefile, indent=2)

#https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
#https://blog.anvetsu.com/posts/custom-dictionary-type-python/
class report_dict(dict):
  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    #self.__dict__[key] = item

  def __getitem__(self, key):
    #obj = self.__dict__[key]
    #obj = super().__getitem__(key)
    #return obj[list(obj.keys())[0]]
    x = self.get(key)
    if (len(x) > 2):
      return self.get(key)
    else:
      obj = super().__getitem__(key)
      return obj[list(obj.keys())[0]]

  def update(self, *args, **kwargs):
    return super().update(*args, **kwargs)
    #return self.__dict__.update(*args, **kwargs)

  def get(self, *args, **kwargs):
    return super().get(*args, **kwargs)

  def keys(self):
    return super().keys()
    #return self.__dict__.keys()

  def values(self):
    return super().values()
    #return self.__dict__.values()

  def items(self):
    return super().items()

  #def pop(self, *args):
  #  return self.__dict__.pop(*args)

  #def __cmp__(self, dict_):
  #  return self.__cmp__(self.__dict__, dict_)

  #def __contains__(self, item):
  #  return item in self.__dict__

  #def __iter__(self):
  #  return iter(self.__dict__)

  #def __unicode__(self):
  #  return unicode(repr(self.__dict__))

#print('Using default dict')
w = {}
l = "Latency"
num = '1'
k1 = { l : num }

w['key1'] = k1

print(w)

k2 = {"Lat" : '2', "inner" : { "Latency" : '2.1' } }

w['key1'].update(k2)
# when updating, the key values in the value needs not be the same

#print(w)
#print(w['key1'])

print("========Custom dict=========")

o = report_dict()

v1 = {"latency" : '1'}

o['key1'] = v1

print(o)
#print(o.items())
#print(o.keys())
#print(o.values())
#print(o['key1'])
#print(o['key1'].get(l))

#o['key1'].update(k2)

v2 = {"Lat" : '2', "inner" : { "Latency" : '2.1' } }
v1.update(v2)

o.update(v1)

print(o)
print('====')
#print(o['key1'])
#print(type(list(o['key1'].values())[1]))
#print(o.get('key1'))
#print(o.get('latency'))

#for key, val in o.items():
  #print(key)
  #print(val)
  #print(val.get('latency'))

def parse_v2(path):
  xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
  if not os.path.isfile(xml_file):
    raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
  
  profile = json.loads(open(os.path.join(path,"report.json"), "r").read())
  clock_unit = profile["UserAssignments"]["unit"]

  out = {}
  #out = report_dict()
  
  summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
  keys = ['TripCount', 'Latency', 'IterationLatency']
  #print(summary.items())
  #print(summary.keys())
  #print(summary.values())
  
  #for v in summary.values():
    #print(v)
    #print(v.get('Latency'))

  def loop(obj, key):
    #if key == 'TripCount':
      #clock_unit = ""
    test = {}
    #test = report_dict()
    for k, v in obj.items():
      #print(k + " -> " + v.get('Latency'))
      val = v.get(key) + " " + clock_unit
      init = { key.lower() : val }
      test[k] = init
      #print(v.items())

      for e, i in v.items():
        if isinstance(i, dict):
          #print(e + " -> " + i.get('Latency'))
          inner_val = i.get(key) + " " + clock_unit
          inner = { e : { key.lower() : inner_val } }
          init.update(inner) # change by custom_dict
          #test[k].update(inner)
          #k.update( (e, i.get('Latency')) )
    return test

  for key in keys:
    out[key] = loop(summary,key)

  #print("=========================")
  #print(out) # checked correct
  #print(out['Latency'])
  #print(out['Latency']['out_matrix_x'])

  testing_self_dict = report_dict()
  testing_self_dict = out
  #print("=========================")
  #print(out['Latency'])

  #print(out['Latency']['out_matrix_x'].keys())
  #interest = out['Latency']['out_matrix_x']
  #print(interest[list(interest.keys())[0]])

  #finalfile = open('final.json', 'w')
  #json.dump(out, finalfile, indent=2)

  #print(out)
  #savefile = open("save.json", "w")
  #json.dump(out, savefile, indent=2)     

parse_v2("./project/")
