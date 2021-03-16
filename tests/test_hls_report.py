import heterocl as hcl
import numpy as np
import re
import json
import xmltodict
import pathlib
import imageio

# TODO: Import once sobel is verified.
def sobel():

    hcl.init(init_dtype=hcl.Float())
  
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + '/test_report_data/rose-grayscale.jpg'  
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

def parse_rpt():
    path = pathlib.Path(__file__).parent.absolute()
    xml_file = str(path) + '/test_report_data/test_csynth.xml'
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
    clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
    summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
  
    info_table = hcl.report.Displayer(clock_unit)
    info_table.get_loops(summary)
    info_table.get_category(summary)
    info_table.scan_range(summary)
    info_table.init_data(summary)
    return info_table

def refine(res_tbl):
    lst = res_tbl.split("\n")
    pattern = re.compile(r'\s\s+') 
    res = []
    for i in lst:
        res.append(re.sub(pattern, ', ', i))
    res[0] = res[0].replace(', ', '', 1)
    return res

def get_expected(wd):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + '/test_report_data/expected.json'
    with open(path) as f:
        data = json.loads(f.read())
    return data[wd]

def test_col(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    res = rpt.display()
    lst = refine(res)
    assert lst[0] == get_expected("Category")

def test_info(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    res = rpt.display()
    lst = refine(res)
    assert lst == get_expected("NoQuery")

def test_loop_query(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    row_query = ['P', 'A']
    lq = rpt.display(loops=row_query)
    lq_lst = refine(lq)
    assert lq_lst == get_expected("LoopQuery")

def test_column_query(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    col_query = ['Trip Count', 'Latency', 'Iteration Latency', 
                 'Pipeline II', 'Pipeline Depth']
    cq = rpt.display(cols=col_query)
    cq_lst = refine(cq)
    assert cq_lst == get_expected("ColumnQuery")

def test_level_query(vhls):
    if vhls:  
        rpt = sobel()
    else:
        rpt = parse_rpt()
    lev_query = 1
    vq = rpt.display(level=lev_query)
    vq_lst = refine(vq)
    assert vq_lst == get_expected("LevelQuery")

def test_level_oob_query(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    lev_query = 5
    vq = rpt.display(level=lev_query)
    vq_lst = refine(vq)
    assert vq_lst == get_expected("LevelQueryOOB")
    lev_query = -2
    try:
        vq = rpt.display(level=lev_query)
    except IndexError:
        assert True
    return

def test_multi_query(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    row_query = ['P', 'A']
    lev_query = 1
    mq = rpt.display(loops=row_query, level=lev_query)
    mq_lst = refine(mq)
    assert mq_lst == get_expected("MultiQuery")

def test_all_query(vhls):
    if vhls:
        rpt = sobel()
    else:
        rpt = parse_rpt()
    row_query = ['P', 'A']
    col_query = ['Latency']
    lev_query = 1
    aq = rpt.display(loops=row_query, level=lev_query, cols=col_query)
    aq_lst = refine(aq)
    assert aq_lst == get_expected("AllQuery")

if __name__ == '__main__':
    test_col()
    test_info()
    test_loop_query()
    test_column_query()
    test_level_query()
    test_level_oob_query()
    test_multi_query()
    test_all_query()
