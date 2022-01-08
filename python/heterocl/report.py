import os, re
import json
import time
import xmltodict
import pandas as pd
# Support for graphical display of the report
#import matplotlib.pyplot as plt
from .report_config import RptSetup
from tabulate import tabulate
from .schedule import Stage

class Displayer(object):
    """
    Queryable report displayer.
  
    ...
  
    Attributes
    ----------
    _category: list
        List of default latency information category.
  
    _category_aux: list
        List of latency information category with range indicators.
  
    _loop_name: list
        List of loop names without loop nest indicators.
  
    _loop_name_aux: list
        List of loop names with loop nest indicators.
  
    _max_level: int
        Maximum level of loop nest in the report file.
  
    _data: dict
        Dictionary containing latency data. 
  
    unit: str
        Unit of each information.
  
    Methods
    ----------
    __checker(lst)
        Checks the validity of frame (unit of report data) list.

    __member_init(elem)
        Extract out properties of the report file.

    __data_acquisition(elem)
        Extract out latency information from the report file.

    init_table(obj)
        Initialize the attributes given the report file.

    collect_data(obj)
        Record latency information provided by the report file.

    get_max(col)
        Sort the latency in a decreasing order for specific latency category.
                                                                      
    display(loops=None, level=None, cols=None)
        Display the report table with appropriate query arguments.
    """

    def __init__(self, unit):
        """
        Parameters
        ----------
        unit: str
            Unit for all numerical values in the table.
        """
        self._category = ['TripCount', 'Latency', 'IterationLatency',
                            'PipelineII', 'PipelineDepth']
        self._category_aux = []
        self._loop_name = []
        self._loop_name_aux = []
        self._max_level = 0
        self._data = {}
        self.unit = unit

    def __is_valid(self, lst):
        """True if the given list of frames with its information are not 
        empty. False otherwise.

        Parameters
        ----------
        lst: list
            List that contains information about the loop in report JSON file
            and its relevant reference information.

        Returns
        ----------
        bool
            Validity of the frame list.
        """
        valid = False
        for elem in lst:
            try:
                if elem[0]:
                    valid |= True
            # Except for the last non-dict case
            except:
                return False
        return valid

    def __member_init(self, elem):
        """Given values to a specific loop, update the class attributes
        accordingly.

        Parameters
        ----------
        elem: tuple
            Tuple containing information about latency values of the current
            loop, the loop name, and other reference information such as the
            current loop level. 

        Returns
        ----------
        list
            List containing information about loops that are next-level down.
        """
        obj, loop, ref, level, loop_aux = elem[0], elem[1], elem[2], elem[3], elem[4]
        
        frame = []
        inner_loops = []
        loop.append(ref)

        if level == 0:
            loop_aux.append(ref)
        else:
            loop_aux.append('+' * level + ' ' + ref)
         
        if len(obj) != 0 :
            for cat in self._category:
                if cat in obj:
                   if isinstance(obj[cat], dict):
                       for index, item in enumerate(self._data):
                           itemlist = list(item)
                           if itemlist[0] == cat:
                               itemlist[1] = 1
                           item = tuple(itemlist)
                           self._data[index] = item
        
        for k in list(obj.keys()):
            if k not in self._category:
                inner_loops.append(k)
    
        for il in inner_loops:
            frame.append((obj[il], loop, il, level+1, loop_aux))
    
        if len(frame) == 0:
            frame.append(({}, loop, ref, level, loop_aux))
    
        return frame
 
    def __data_acquisition(self, elem):
        """From latency values in a specific loop, extract out the latency 
        values.
                                                                 
        Parameters
        ----------
        elem: tuple
            Single-element tuple containing information about latency values 
            of the current loop. 
                   
        Returns
        ----------
        list                                                                         
            List containing information about loops that are next-level down.
        """ 
        obj, val_dict = elem[0], elem[1]
        
        frame = []
        inner_loops = []
    
        if len(obj) != 0:
            for cat in self._category_aux:
                cat_split = cat.split(' ', 1)
                val = 'N/A'

                key = cat.replace(' ', '')
                # Not a min-max value
                if key in self._category:
                   try:
                       val = obj[key]
                   except:
                       pass
                else:
                   cat_split = cat.split(' ', 1)
                   minmax = cat_split[0].lower()
                   key = cat_split[1].replace(' ', '')
                   if isinstance( obj[key], dict ):
                       val = obj[key]['range'][minmax]
                   else:
                       val = obj[key]
                
                val_dict[cat] = val
        for s in list(obj.keys()):
            if s not in self._category:
                inner_loops.append(s)
                                                                 
        for il in inner_loops:
            frame.append((obj[il], val_dict))
                                                                 
        if len(frame) == 0:
            frame.append(({}, val_dict))
 
        return frame

    def init_table(self, obj):
        """Initialize attributes defined above for the specific report file.

        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.       

        Returns
        ----------
        None
        """
        keys = list(obj.keys())
    
        def extract_category(elem):
            obj, level, cat_lst = elem[0], elem[1], elem[2]
            frame = []
            inner_loops = []

            for key in obj.keys():
                if not isinstance(obj[key], dict) or ('range' in list(obj[key].keys())):
                    if key not in cat_lst:
                        cat_lst.append(key)
                else:
                    inner_loops.append(key)

            for il in inner_loops:
                frame.append((obj[il], level+1, cat_lst))

            if len(frame) == 0:
                frame.append(({}, level, cat_lst))
            return frame

        lst = []
        cat_lst = []
        for k in keys:
            lst.append((obj[k], 0, cat_lst))

        while self.__is_valid(lst):
            lst = list(map(extract_category, lst))
            lst = [item for elem in lst for item in elem]

        accum = []
        for elem in lst:
            accum.append(elem[2])

        max_len = max(map(len, accum))
        res = []
        for i in range(max_len):
            for elem in accum:
                try:
                    if elem[i] not in res:
                        res.append(elem[i])
                except:
                    pass

        if "PipelineII" not in res:
            res.append("PipelineII")
        if "PipelineDepth" not in res:
            res.append("PipelineDepth")

        self._category = res

        frame_lst = []

        for k in keys:
            frame_lst.append((obj[k], [], k, 0, [])) 
        
        # Temporarily make use of data to be a list
        self._data = []
        for cat in self._category:
            self._data.append((cat, 0))

        while self.__is_valid(frame_lst):
            frame_lst = list(map(self.__member_init, frame_lst))
    
            frame_lst = [item for elem in frame_lst for item in elem]
        
        self._max_level = max([x[3] for x in frame_lst])
    
        filtered = [x[1] for x in frame_lst]
        filtered_aux = [x[4] for x in frame_lst]
        
        self._loop_name = [item for elem in filtered for item in elem]
        self._loop_name = list(dict.fromkeys(self._loop_name))
    
        self._loop_name_aux = [item for elem in filtered_aux for item in elem]
        self._loop_name_aux = list(dict.fromkeys(self._loop_name_aux))
                                                                               
        for cat, r in self._data:
            data_cat = re.sub(r"(\w)([A-Z])", r"\1 \2", cat)
            if r == 0:
                self._category_aux.append(data_cat)
            else:
                self._category_aux.append('Min ' + data_cat)
                self._category_aux.append('Max ' + data_cat) 
        
        # Re-initialize the data field
        self._data = {}
  
    def collect_data(self, obj):
        """Collect latency data from the given report file.
                   
        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.                          

        Returns
        ----------
        None
        """
        keys = list(obj.keys())
                                                                      
        frame_lst = []

        fin_dict = {key: [] for key in self._category_aux}
        self._data = {key: [] for key in self._category_aux}

        for k in keys:
            frame_lst.append((obj[k], {}))

        while self.__is_valid(frame_lst):
            frame_lst = list(map(self.__data_acquisition, frame_lst))
 
            for cat in self._category_aux:
                store = []
                for elem in frame_lst:
                    try:
                        store.append(elem[0][1][cat])
                    except:
                        pass
                fin_dict[cat].append(store)

            new_frame_lst = []  
            for elem in frame_lst:
                for item in elem:                    
                    new_frame_lst.append((item[0], {}))

            frame_lst = new_frame_lst

        lev_seq = list(map(lambda x: x.count('+'), self._loop_name_aux))
        for cat in self._category_aux:
            for lev in lev_seq:
                self._data[cat].append(fin_dict[cat][lev].pop(0))

    def get_max(self, col):
        """Form a 3-element tuple list that sorts loops in a decreasing order
        with respect to the latency information of the specified latency 
        category.
                   
        Parameters
        ----------
        col: str
            Latency category name.
                   
        Returns
        ----------
        list
            3-element tuple list with loop names, its data corresponding to
            [col] latency category, and the loop level.
        """
        tup_lst = list(map(lambda x, y, z: (x, y, z), self._loop_name, self._data[col], self._loop_name_aux))
        tup_lst = list(map(lambda x: (x[0], x[1], x[2].count('+')), tup_lst))
        return list(reversed(sorted(tup_lst, key=lambda x: int(x[1]))))

    def display(self, loops=None, level=None, cols=None):
        """Display the report file.
  
        Parameters
        ----------
        loops: list, optional
            List of loop names (e.g., ['A', 'Y'])
        level: int, optional
            Maximum level of loop nest to print.
        cols: list, optional
            List of column names. (e.g., ['Trip Count'])
   
        Returns
        ----------
        str
            String representation of pandas dataframe being displayed.
        """
        if loops is None:
            loops = self._loop_name_aux
        if level is None:
            level = self._max_level
        if cols is None:
            cols = self._category_aux
  
        selected = []
        for l in loops:
            if type(l) != str:
                l = str(l).split(",")[0].split("(")[1]
                # TODO: add support for axis value specification

            for k in self._loop_name_aux:
                if l in k:
                    selected.append(k) 
        
        rows = []
        if level > self._max_level:
            rows = selected
        else:
          for k in selected:
              lev = k.count('+')
              if lev <= level:
                  rows.append(k)

        ncols = []
        for c in cols:
            for ca in self._category_aux:
                if c in ca:
                    ncols.append(ca)

        alignment = ('left',)
        for i in range(len(cols)):
            alignment = alignment + ('right',)

        df = pd.DataFrame(data=self._data, index=self._loop_name_aux)
        print(tabulate(df.loc[rows, cols], headers=cols, tablefmt='psql', colalign=alignment))
        print('* Units in {}'.format(self.unit))
        splt = df.loc[rows, cols].to_string().split("\n")
        pd.set_option('max_colwidth', len(splt[0]) * 100)
        return df.loc[rows, cols].to_string()

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
    

def parse_xml(path, xml_path, prod_name, print_flag=False):
    xml_file = os.path.join(path, xml_path)

    if not os.path.isfile(xml_file):
        raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
    json_file = os.path.join(path,"report.json")
    outfile = open(json_file, "w")
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
        json.dump(profile, outfile, indent=2)

    config = RptSetup(profile, prod_name)
    config.eval_members()

    res = {}
    res["HLS Version"] = config.prod_name + " " + config.version
    res["Product family"] = config.prod_family
    res["Target device"] = config.target_device
    res["Top Model Name"] = config.top_model_name
    res["Target CP"] = config.target_cp + " " + config.assignment_unit
    res["Estimated CP"] = config.estimated_cp + " " + config.assignment_unit
    res["Latency (cycles)"] = "Min {:<6}; ".format(config.min_latency) + \
                              "Max {:<6}".format(config.max_latency)
    res["Interval (cycles)"] = "Min {:<6}; ".format(config.min_interval) + \
                               "Max {:<6}".format(config.max_interval)

    est_resources = config.est_resources
    avail_resources = config.avail_resources
    key_avail = list(avail_resources.keys())

    resources = {}
    for name in key_avail:
        try:
            item = [est_resources[name], avail_resources[name]]
            item.append("{}%".format(round(int(item[0])/int(item[1])*100)))
            resources[name] = item.copy()
        except ZeroDivisionError:
            item.append("0%")
            resources[name] = item.copy()
        except:
            pass
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
    clock_unit = config.performance_unit
    summary = config.loop_latency

    info_table = Displayer(clock_unit)
    info_table.init_table(summary)
    info_table.collect_data(summary)

    if print_flag:
        print(table)
    return info_table

def report_stats(target, folder):
    path = folder

    file_dir = []
    for root, _, files in os.walk(path):
        if "test_csynth.xml" in files:
            file_dir.append(os.path.join(root, "test_csynth.xml"))
    dirs = file_dir[0]

    xml_path = dirs.split('/', 1)[1]

    # If report file is not found, error out.
    if not xml_path:
        raise RuntimeError("Not found report statistics")

    proj_path = dirs.split('/')[1]

    if target.tool.name == "vivado_hls":
        if os.path.isdir(os.path.join(path, proj_path)):
            return parse_xml(path, xml_path, "Vivado HLS")
        else:
            raise RuntimeError("Not found %s folder" % proj_path)

    elif target.tool.name == "aocl":
        if os.path.isdir(os.path.join(path, "kernel/reports")):
            return parse_js(path)

    elif target.tool.name == "vitis":
        if os.path.isdir(os.path.join(path, proj_path)):
            return parse_xml(path, xml_path, "Vitis HLS", True)
        else:
            raise RuntimeError("Not found %s folder" % proj_path)

    else:
        raise RuntimeError("tool {} not yet supported".format(target.tool.name))
