import heterocl as hcl
import re
import json
import xmltodict
import sys
sys.path.append('../python/heterocl')
from rptdisp import RptDisp

def parse_rpt():
  xml_file = './test_report_data/test_csynth.xml'
  with open(xml_file, "r") as xml:
    profile = xmltodict.parse(xml.read())["profile"]
  clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
  summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]

  info_table = RptDisp(clock_unit)
  info_table.scan_range(summary)
  info_table.get_loops(summary)
  info_table.init_data(summary)
  return info_table

def refine(lst):
  pattern = re.compile(r'\s\s+') 
  res = []
  for i in lst:
    res.append(re.sub(pattern, ', ', i))
  res[0] = res[0].replace(', ', '', 1)
  return res

def get_expected(wd):
  with open('./test_report_data/expected.json') as f:
    data = json.loads(f.read())
  return data[wd]

def test_info(rpt):
  res = rpt.display()
  res_str = res.split("\n") 
  lst = refine(res_str)
  assert lst == get_expected("NoQuery")

def test_loop_query(rpt):
  row_query = ['P', 'A']
  lq = rpt.display( loops=row_query )
  lq_str = lq.split("\n") 
  lq_lst = refine(lq_str)
  assert lq_lst == get_expected("LoopQuery")

def test_column_query(rpt):
  col_query = ['Latency']
  cq = rpt.display( cols=col_query )
  cq_str = cq.split("\n")
  cq_lst = refine(cq_str)
  assert cq_lst == get_expected("ColumnQuery")

def test_level_query(rpt):
  lev_query = 1
  vq = rpt.display( level=lev_query )
  vq_str = vq.split("\n")
  vq_lst = refine(vq_str)
  assert vq_lst == get_expected("LevelQuery")

def test_multi_query(rpt):
  row_query = ['P', 'A']
  lev_query = 1
  mq = rpt.display( loops=row_query, level=lev_query )
  mq_str = mq.split("\n")
  mq_lst = refine(mq_str)
  assert mq_lst == get_expected("MultiQuery")

if __name__ == '__main__':
  rpt = parse_rpt()
  test_info(rpt)
  test_loop_query(rpt)
  test_column_query(rpt)
  test_level_query(rpt)
  test_multi_query(rpt)
