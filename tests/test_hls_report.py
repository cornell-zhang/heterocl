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
  
    target = hcl.Platform.xilinx_zc706  
    target.config(compiler="vivado_hls", mode="csyn")
  
    hcl_img = hcl.asarray(img)
    hcl_Gx = hcl.asarray(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
    hcl_Gy = hcl.asarray(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    hcl_F = hcl.asarray(np.zeros((height, width)))
  
    f = hcl.build(s,target) 
    f(hcl_img, hcl_Gx, hcl_Gy, hcl_F)
    return f.report()

# TODO: Import once algorithms are stabilized.
def canny():
    pass

def spam_filter():
    pass
# END TODO

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

def get_rpt(config):
    vhls = config['vhls']

    if vhls and config['has_algorithm']:
        rpt = eval(alg['name'] + '()')
    else:
        path = pathlib.Path(__file__).parent.absolute()
        xml_file = str(path) + config['algorithm']['report_path']
        with open(xml_file, "r") as xml:
            profile = xmltodict.parse(xml.read())["profile"]
        clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
        summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
  
        rpt = hcl.report.Displayer(clock_unit)
        rpt.init_table(summary)
        rpt.collect_data(summary)
    return rpt

def _test_rpt(config):
    
    alg_name = config['algorithm']['name']
    rpt = get_rpt(config)

    def test_get_max():
        res = rpt.get_max(config['get_max'])
        res_dict = {x : {y : z} for x, y, z in res} 
        assert res_dict == get_expected(alg_name, 'GetMax')

    def test_col(): 
        res = rpt.display()
        lst = refine(res)
        assert lst[0] == get_expected(alg_name, config['col'])

    def test_info():
        rpt = get_rpt(config)
        res = rpt.display()
        lst = refine(res)
        assert lst == get_expected(alg_name, config['info'])

    def test_loop_query(): 
        loop_query = config['loop_query']
        row_query = loop_query['query']
        res = rpt.display(loops=row_query)
        lst = refine(res)
        assert lst == get_expected(alg_name, loop_query['name'])

    def test_column_query():
        column_query = config['column_query']
        col_query = column_query['query']
        res = rpt.display(cols=col_query)
        lst = refine(res)
        assert lst == get_expected(alg_name, column_query['name'])

    def test_level_query():
        level_query = config['level_query']
        res = rpt.display(level=level_query['val'])
        lst = refine(res)
        assert lst == get_expected(alg_name, level_query['name'])

    def test_level_oob_query():
        level_out_of_bound = config['level_out_of_bound']
        res = rpt.display(level=level_out_of_bound['val'][0])
        lst = refine(res)
        assert lst == get_expected(alg_name, level_out_of_bound['name'])
    
        try:
            res = rpt.display(level=level_out_of_bound['val'][1])
        except IndexError:
            assert True
        return

    def test_multi_query():
        multi_query = config['multi_query']
        row_query = multi_query['row_query']
        lev_query = multi_query['level_query']
        res = rpt.display(loops=row_query, level=lev_query)
        lst = refine(res)
        assert lst == get_expected(alg_name, multi_query['name'])

    def test_all_query():
        all_query = config['all_query']
        row_query = all_query['row_query']
        col_query = all_query['col_query']
        lev_query = all_query['level_query']
        res = rpt.display(loops=row_query, level=lev_query, cols=col_query)
        lst = refine(res)
        assert lst == get_expected(alg_name, all_query['name'])

    test_get_max()
    test_col()
    test_info()
    test_loop_query()
    test_column_query()
    test_level_query()
    test_level_oob_query()
    test_multi_query()
    test_all_query()

def test_knn_digitrec(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 1,
        'algorithm' : {
            'report_path' : '/test_report_data/digitrec_report.xml',
            'name' : 'knn_digitrec'
        },
        'get_max' : 'Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['knn_update'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Trip Count'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 0,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['knn_update_y1'],
            'level_query' : 0,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['knn_mat_burst_s0_knn_mat_burst_s1'],
            'col_query' : ['Latency'],
            'level_query' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_kmeans(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 1,
        'algorithm' : {
            'report_path' : '/test_report_data/kmeans_report.xml',
            'name' : 'kmeans'
        },
        'get_max' : 'Absolute Time Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['points_burst'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Latency'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 0,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['main_loop'],
            'level_query' : 1,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['calc_sum'],
            'col_query' : ['Latency'],
            'level_query' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_sobel(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 0,
        'algorithm' : {
            'report_path' : '/test_report_data/sobel_report.xml',
            'name' : 'sobel'
        },
        'get_max' : 'Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['P', 'A'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Trip Count', 'Latency', 'Iteration Latency', 
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 1,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['P', 'A'],
            'level_query' : 1,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['P', 'A'],
            'col_query' : ['Latency'],
            'level_query' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_sobel_partial(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 0,
        'algorithm' : {
            'report_path' : '/test_report_data/sobel_report_partial.xml',
            'name' : 'sobel_partial'
        },
        'get_max' : 'Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['B', 'D'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Trip Count', 'Latency', 'Iteration Latency', 
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 2,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['B', 'D'],
            'level_query' : 1,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['B', 'D'],
            'col_query' : ['Trip Count'],
            'level_query' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_canny(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 0,
        'algorithm' : {
            'report_path' : '/test_report_data/canny_report.xml',
            'name' : 'canny'
        },
        'get_max' : 'Max Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['A', 'Y'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Trip Count', 'Min Latency', 'Max Latency', 
                        'Min Iteration Latency', 'Max Iteration Latency',
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 2,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['A', 'Y'],
            'level_query' : 1,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['A', 'Y'],
            'col_query' : ['Max Latency'],
            'level_query' : 3,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

def test_spam_filter(vhls):
    config = {
        'vhls' : vhls,
        'has_algorithm' : 0,
        'algorithm' : {
            'report_path' : '/test_report_data/spam_filter_report.xml',
            'name' : 'spam_filter'
        },
        'get_max' : 'Latency',
        'col' : 'Category',
        'info' : 'NoQuery',
        'loop_query' : {
            'query' : ['loop_x'],
            'name' : 'LoopQuery'
        },
        'column_query' : {
            'query' : ['Trip Count', 'Latency', 'Iteration Latency', 
                        'Pipeline II', 'Pipeline Depth'],
            'name' : 'ColumnQuery'
        },
        'level_query' : {
            'val' : 0,
            'name' : 'LevelQuery'
        },
        'level_out_of_bound' : {
            'val' : [5, -2],
            'name' : 'LevelQueryOOB'
        },
        'multi_query' : {
            'row_query' : ['update_param'],
            'level_query' : 1,
            'name' : 'MultiQuery'
        },
        'all_query' : {
            'row_query' : ['dot_product'],
            'col_query' : ['Latency'],
            'level_query' : 1,
            'name' : 'AllQuery'
        }
    }
    _test_rpt(config)

if __name__ == '__main__':
    test_knn_digitrec(False)
    test_kmeans(False)
    test_sobel(False)
    test_sobel_partial(False)
    test_canny(False)
    test_spam_filter(False) 
