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

# TODO: Import once spam_filter algorithm is stabilized.
def spam_filter():
    pass

def refine(res_tbl):
    lst = res_tbl.split("\n")
    pattern = re.compile(r'\s\s+') 
    res = []
    for i in lst:
        res.append(re.sub(pattern, ', ', i))
    res[0] = res[0].replace(', ', '', 1)
    return res

def get_expected(ver, wd):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + '/test_report_data/expected.json'
    with open(path) as f:
        data = json.loads(f.read())
    return data[ver][wd]

# Example 1: Sobel Edge Detection algorithm

def parse_rpt():
    path = pathlib.Path(__file__).parent.absolute()
    xml_file = str(path) + '/test_report_data/test_csynth.xml'
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
    clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
    summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
  
    info_table = hcl.report.Displayer(clock_unit)
    info_table.init_table(summary)
    info_table.collect_data(summary)
    return info_table

# Example 2: Spam email filter algorithm

def parse_rpt_same():
    path = pathlib.Path(__file__).parent.absolute()
    xml_file = str(path) + '/test_report_data/report.xml'
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
    clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
    summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
                                                                                    
    info_table = hcl.report.Displayer(clock_unit)
    info_table.init_table(summary)
    info_table.collect_data(summary)
    return info_table

# Test Suite

def _test_rpt(config):
    def test_col(config):
        vhls = config['vhls']
        ver = config['ver']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        res = rpt.display()
        lst = refine(res)
        assert lst[0] == get_expected(ver, config['col'])

    def test_info(config):
        vhls = config['vhls']
        ver = config['ver']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        res = rpt.display()
        lst = refine(res)
        assert lst == get_expected(ver, config['info'])

    def test_loop_query(config):
        vhls = config['vhls']
        ver = config['ver']
        lpq = config['lpq']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        row_query = lpq['query']
        res = rpt.display(loops=row_query)
        lst = refine(res)
        assert lst == get_expected(ver, lpq['name'])

    def test_column_query(config):
        vhls = config['vhls']
        ver = config['ver']
        cq = config['cq']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        col_query = cq['query']
        res = rpt.display(cols=col_query)
        lst = refine(res)
        assert lst == get_expected(ver, cq['name'])

    def test_level_query(config):
        vhls = config['vhls'] 
        ver = config['ver']
        lev = config['lev']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        res = rpt.display(level=lev['val'])
        lst = refine(res)
        assert lst == get_expected(ver, lev['name'])

    def test_level_oob_query(config):
        vhls = config['vhls']
        ver = config['ver']
        oob = config['oob']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        res = rpt.display(level=oob['val'][0])
        lst = refine(res)
        assert lst == get_expected(ver, oob['name'])
    
        try:
            res = rpt.display(level=oob['val'][1])
        except IndexError:
            assert True
        return

    def test_multi_query(config):
        vhls = config['vhls']
        ver = config['ver']
        mq = config['mq']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        row_query = mq['row']
        lev_query = mq['lev']
        res = rpt.display(loops=row_query, level=lev_query)
        lst = refine(res)
        assert lst == get_expected(ver, mq['name'])

    def test_all_query(config):
        vhls = config['vhls']
        ver = config['ver']
        aq = config['aq']
    
        if vhls and config['has_algorithm']:
            rpt = eval(config['algorithm'] + '()')
        else:
            if ver == 'Normal':
                rpt = parse_rpt()
            else:
                rpt = parse_rpt_same()
        
        row_query = aq['row']
        col_query = aq['col']
        lev_query = aq['lev']
        res = rpt.display(loops=row_query, level=lev_query, cols=col_query)
        lst = refine(res)
        assert lst == get_expected(ver, aq['name'])

    test_col(config)
    test_info(config)
    test_loop_query(config)
    test_column_query(config)
    test_level_query(config)
    test_level_oob_query(config)
    test_multi_query(config)
    test_all_query(config)

def test_sobel():
    config = {
        'ver' : 'Normal',
        'vhls' : False,
        'has_algorithm' : 0,
        'algorithm' : 'sobel', 
        'col' : 'Category',
        'info' : 'NoQuery',
        'lpq' : {
            'query' : ['P', 'A'],
            'name' : 'LoopQuery'
        },
        'cq' : {
            'query' : ['Trip Count', 'Latency', 'Iteration Latency', 
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'lev' : {
            'val' : 1,
            'name' : 'LevelQuery'
        },
        'oob' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'mq' : {
            'row' : ['P', 'A'],
            'lev' : 1,
            'name' : 'MultiQuery'
        },
        'aq' : {
            'row' : ['P', 'A'],
            'col' : ['Latency'],
            'lev' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_spam_filter():
    config = {
        'ver' : 'Same',
        'vhls' : False,
        'has_algorithm' : 0,
        'algorithm' : 'spam_filter', 
        'col' : 'Category',
        'info' : 'NoQuery',
        'lpq' : {
            'query' : ['loop_x'],
            'name' : 'LoopQuery'
        },
        'cq' : {
            'query' : ['Trip Count', 'Latency', 'Iteration Latency', 
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'lev' : {
            'val' : 0,
            'name' : 'LevelQuery'
        },
        'oob' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'mq' : {
            'row' : ['update_param'],
            'lev' : 1,
            'name' : 'MultiQuery'
        },
        'aq' : {
            'row' : ['dot_product'],
            'col' : ['Latency'],
            'lev' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

if __name__ == '__main__':
    test_sobel()
    test_spam_filter()  
