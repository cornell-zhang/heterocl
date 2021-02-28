import heterocl as hcl
import numpy as np
import imageio
import re
import json

def sobel():

  hcl.init(init_dtype=hcl.Float())

  path = './test_report_data/rose-grayscale.jpg'
  img = imageio.imread(path)

  height, width, rgb = img.shape

  imgF = hcl.placeholder((height, width,3), "Image")
  Gx = hcl.placeholder((3,3), "Gx")
  Gy = hcl.placeholder((3,3), "Gy")
 
  def sobel_kernel(imgF, Gx, Gy):
    def pad(x,y,z):
      out = hcl.scalar(0, "out")
      with hcl.if_(hcl.and_(x > 0, y > 0)):
        out.v = imgF[x-1,y-1,z]
      with hcl.else_():
        out.v = 0
      return out.v
    P = hcl.compute((height+2, width+2, 3), lambda x,y,z: pad(x,y,z), "P")

    A = hcl.compute((height+2, width+2), lambda x,y: P[x][y][0] + P[x][y][1] + P[x][y][2], "A") 

    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    resX = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "X")

    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    resY = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum2"), "Y")

    R = hcl.compute((height, width), lambda x,y: hcl.sqrt(resX[x][y]*resX[x][y] + resY[x][y]*resY[x][y]), "R")
  
    norm = hcl.scalar(255/4328)

    return hcl.compute((height, width), lambda x,y: R[x][y] * norm.v, "F")

  s = hcl.create_schedule([imgF, Gx, Gy], sobel_kernel)

  sA = sobel_kernel.A
  sX = sobel_kernel.X
  sY = sobel_kernel.Y

  LBX = s.reuse_at(sA._op, s[sX], sX.axis[0], "LBX")
  LBY = s.reuse_at(sA._op, s[sY], sY.axis[0], "LBY")
  WBX = s.reuse_at(LBX, s[sX], sX.axis[1], "WBX")
  WBY = s.reuse_at(LBY, s[sY], sY.axis[1], "WBY")
  s.partition(LBX, dim=1)
  s.partition(LBY, dim=1)
  s.partition(WBX)
  s.partition(WBY)
  s.partition(Gx)
  s.partition(Gy)
  sP = sobel_kernel.P
  sR = sobel_kernel.R
  sF = sobel_kernel.F
  s[sX].pipeline(sX.axis[1])
  s[sY].pipeline(sY.axis[1])
  s[sR].pipeline(sR.axis[1])
  s[sF].pipeline(sF.axis[1])
  
  target = hcl.platform.zc706  
  target.config(compile="vivado_hls", mode="csyn")

  hcl_img = hcl.asarray(img)
  hcl_Gx = hcl.asarray(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
  hcl_Gy = hcl.asarray(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
  hcl_F = hcl.asarray(np.zeros((height, width)))

  f = hcl.build(s,target) 
  f(hcl_img, hcl_Gx, hcl_Gy, hcl_F)
  return f.report()

def refine(lst):
  pattern = re.compile(r'\s\s+') 
  res = []
  for i in lst:
    res.append(re.sub(pattern, ', ', i))
  res[0] = res[0].replace(', ', '', 1)
  return res

def test_info(expected):
  rpt = sobel()
  res = rpt.display()
  res_str = res.split("\n") 
  lst = refine(res_str)
  assert lst == expected

def test_loop_query(expected):
  rpt = sobel()
  row_query = ['P', 'A']
  lq = rpt.display( loops=row_query )
  lq_str = lq.split("\n") 
  lq_lst = refine(lq_str)
  assert lq_lst == expected

def test_column_query(expected):
  rpt = sobel()
  col_query = ['Latency']
  cq = rpt.display( cols=col_query )
  cq_str = cq.split("\n")
  cq_lst = refine(cq_str)
  assert cq_lst == expected

def test_level_query(expected):
  rpt = sobel()
  lev_query = 1
  vq = rpt.display( level=lev_query )
  vq_str = vq.split("\n")
  vq_lst = refine(vq_str)
  assert vq_lst == expected

def test_multi_query(expected):
  rpt = sobel()  
  row_query = ['P', 'A']
  lev_query = 1
  mq = rpt.display( loops=row_query, level=lev_query )
  mq_str = mq.split("\n")
  mq_lst = refine(mq_str)
  assert mq_lst == expected

if __name__ == '__main__':
  with open('./test_report_data/expected.json') as f:
    data = json.loads(f.read())
  test_info(data["NoQuery"])
  test_loop_query(data["LoopQuery"])
  test_column_query(data["ColumnQuery"])
  test_level_query(data["LevelQuery"])
  test_multi_query(data["MultiQuery"])
