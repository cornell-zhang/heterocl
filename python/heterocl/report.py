import os, re
import json
import time
import xmltodict
from tabulate import tabulate
import pandas as pd

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
    __get_value(v, key, minmax)
        Get the value associated with the input key.
  
    __info_extract(obj, key, minmax, col)
        Extract out all the latency information from the report. 

    get_loops(obj)
        Acquire loop names with and without loop nest indicators.

    get_category(obj)
        Scans the parsed xml file to check what latency categories
        there are.

    scan_range(obj)
        Scan the entire report file to see which latency category contains
        range values.
    
    init_data(obj)
        Initialize the data given the report file.  
                                                                      
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

    def __checker(self, lst):
        valid = False
        for elem in lst:
            if elem[0]:
                valid |= True
        return valid 

    def __member_init(self, elem):
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
                data_cat = re.sub(r"(\w)([A-Z])", r"\1 \2", cat)
                already_in = False
    
                for k in list(self._data.keys()):
                    if data_cat in k:
                        already_in = True
    
                if not already_in:
                    self._data[data_cat] = []
    
                if cat in obj:
                    val = obj[cat]
                    if isinstance( val, dict ):
                        self._data.popitem()
                        self._data['Min ' + data_cat] = []
                        self._data['Max ' + data_cat] = []
        
        for k in list(obj.keys()):
            if k not in self._category:
                inner_loops.append(k)
    
        for il in inner_loops:
            frame.append((obj[il], loop, il, level+1, loop_aux))
    
        if len(frame) == 0:
            frame.append(({}, loop, ref, level, loop_aux))
    
        return frame
    
    def init_table(self, obj):
        keys = list(obj.keys())
    
        frame_lst = []
    
        for k in keys:
            #             frame, loop, name, level, loop_aux
            frame_lst.append((obj[k], [], k, 0, []))
    
        correct = self.__checker(frame_lst)
        
        while correct:
            frame_lst = list(map(self.__member_init, frame_lst))
    
            frame_lst = [item for elem in frame_lst for item in elem]
    
            correct = self.__checker(frame_lst)
    
        filtered = [x[1] for x in frame_lst]
        filtered_aux = [x[4] for x in frame_lst]
        
        self._loop_name = [item for elem in filtered for item in elem]
        self._loop_name = list(dict.fromkeys(self._loop_name))
    
        self._loop_name_aux = [item for elem in filtered_aux for item in elem]
        self._loop_name_aux = list(dict.fromkeys(self._loop_name_aux))

        self._category_aux = list(self._data.keys())
    
    def data_acquisition(self, elem):
    
        obj, loop, ref = elem[0], elem[1], elem[2]
        
        frame = []
        inner_loops = []
    
        if len(obj) != 0:
            for cat in self._category_aux:
                cat_split = cat.split(' ', 1)
                val = 'N/A'
    
                if len(cat_split) > 1:
                    k = cat_split[1].replace(' ', '')
                    minmax = cat_split[0].lower()
                    if minmax != 'min' and minmax != 'max':
                        key = cat.replace(' ', '')
                        if key in obj:
                            val = obj[key]
                    else:
                        val = obj[k]
                        if isinstance( val, dict ):
                            val = obj[k]['range'][minmax]
                else:
                    val = obj[cat]
    
                self._data[cat].append(val)          
    
        for s in list(obj.keys()):
            if s not in self._category:
                inner_loops.append(s)
                                                                 
        for il in inner_loops:
            frame.append((obj[il], loop, il))
                                                                 
        if len(frame) == 0:
            frame.append(({}, loop, ref))
                                                                 
        return frame
    
    def collect_data(self, obj):
        keys = list(obj.keys())
                                                                      
        frame_lst = []
                                                                      
        for k in keys:
            #             frame, loop, name
            frame_lst.append((obj[k], [], k))
                                                                      
        correct = self.__checker(frame_lst)
        
        while correct:
            frame_lst = list(map(self.data_acquisition, frame_lst))
                                                                      
            frame_lst = [item for elem in frame_lst for item in elem]
                                                                      
            correct = self.__checker(frame_lst)

    def __get_value(self, v, key, minmax):
        """Gets the value associated with _key_. If the value is a range
        value, get the appropriate 'min' or 'max' value, determined by
        _minmax_.
  
        Parameters
        ----------
        v: dict
            Dictionary containing all latency information for a particular loop.
        key: str
            Latency category.
        minmax: str
            Range indicator (min or max).
   
        Returns
        ----------
        str
            Latency value of the loop with category 'key'. 
        """
        num = v.get(key)
        val = 'N/A'
        if isinstance(num, str):
            val = str(num)
        elif isinstance(num, dict):
            val = num['range'][minmax]
        return val
  
    def __info_extract(self, obj, key, minmax, col):
        """Extract out all the latency information from the report.
  
        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.
        key: str
            Latency category. 
        minmax: str
            Range indicator (min or max). 
        col: list
            Column name in the data.
        
        Returns
        ----------
        None
        """
        for k, v in obj.items():      
            val = self.__get_value(v, key, minmax)
            self._data[col].append(val)
            in_k, in_v = list(v.items())[-1]
            while not isinstance(in_v, str):       
                val = self.__get_value(in_v, key, minmax)
                self._data[col].append(val)
                in_k, in_v = list(in_v.items())[-1]

    def get_loops(self, obj):
        """Initializes the loop name lists.
                                                           
        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file. 
                                                           
        Returns
        ----------
        None
        """
        for k, v in obj.items():
            self._loop_name.append(k)
            self._loop_name_aux.append(k)
            in_k, in_v = list(v.items())[-1]
            n = 0
            while not isinstance(in_v, str):
                n = n + 1
                k = '+' * n + ' ' + in_k
                self._loop_name.append(in_k)
                self._loop_name_aux.append(k)
                in_k, in_v = list(in_v.items())[-1]
            if (n > self._max_level):
                self._max_level = n

    def get_category(self, obj):
        """Scans the parsed xml file to check what latency categories
        there are.

        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.

        Returns
        ----------
        None
        """
        cat_lst = []
        for k, v in obj.items():
            cat_lst = cat_lst + list(v.keys())
            in_k, in_v = list(v.items())[-1]
            while not isinstance(in_v, str):
                cat_lst = cat_lst + list(in_v.keys())
                in_k, in_v = list(in_v.items())[-1]
        simpl_lst = [i for n, i in enumerate(cat_lst) if i not in cat_lst[:n]]
        res = []
        for cat in simpl_lst:
            if cat not in self._loop_name:
                re_outer = re.compile(r'([^A-Z ])([A-Z])')
                re_inner = re.compile(r'(?<!^)([A-Z])([^A-Z])')
                res.append(re_outer.sub(r'\1 \2', re_inner.sub(r' \1\2', cat)))
        self._category = res
     
    def scan_range(self, obj):
        """Scans the parsed xml file to check which categories have range 
        values and updates _category_aux accordingly. Also, it initializes
        _data to be used in displaying the report data.
  
        Parameters
        ----------
        obj: dict 
            Dictionary representation of the report file.
  
        Returns
        ----------
        None
        """
        detect_minmax = []
        for item in self._category:
            cat = item.replace(' ', '')
            has_minmax = False
            for k, v in obj.items():
                has_minmax = has_minmax or isinstance(v.get(cat), dict)
                in_k, in_v = list(v.items())[-1]
                while not isinstance(in_v, str):
                    has_minmax = has_minmax or isinstance(v.get(cat), dict)
                    in_k, in_v = list(in_v.items())[-1]
  
            if has_minmax:
                detect_minmax.append('Min ' + item)
                detect_minmax.append('Max ' + item)
            else:
                detect_minmax.append(item)
  
        self._category_aux = detect_minmax
        for c in self._category_aux:
            self._data[c] = []

    def init_data(self, obj):
        """Initialize the _data attribute.
   
        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file. 
        
        Returns
        ----------
        None
        """
        for col in self._category_aux:
            key_split = col.split(' ', 1)
            if len(key_split) > 1:
                key        = key_split[1].replace(' ', '')
                minmax     = key_split[0].lower()
                info_tuple = (key, minmax)
                if minmax != 'min' and minmax != 'max':
                    info_tuple = (col.replace(' ', ''), '')
            else:
                info_tuple = (col.replace(' ', ''), '') 
            self.__info_extract(obj, info_tuple[0], info_tuple[1], col)    
    
    def get_max(self, col):
        tup_lst = list(map(lambda x, y: (x, y), self._loop_name, self._data[col]))
        return sorted(tup_lst, key=lambda x: x[1])
    
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
    

def parse_xml(path, print_flag=False):
    xml_file = os.path.join(path, "out.prj", "solution1/syn/report/test_csynth.xml")
    if not os.path.isfile(xml_file):
        raise RuntimeError("Cannot find {}, run csyn first".format(xml_file))
    json_file = os.path.join(path,"report.json")
    outfile = open(json_file, "w")
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
        json.dump(profile, outfile, indent=2)

    #names = json.load(open('config_report.json',))

    user_assignment = profile["UserAssignments"]
    perf_estimate = profile["PerformanceEstimates"]
    area_estimate = profile["AreaEstimates"]
    overall_latency = perf_estimate["SummaryOfOverallLatency"]
    #user_assignment = profile[names["Estimates"]["Assignments"]]
    #perf_estimate = profile[names["Estimates"]["Performance"]]
    #area_estimate = profile[names["Estimates"]["Area"]]
    #overall_latency = perf_estimate[names["Latency"]["Name"]]

    res = {}
    res["HLS Version"] = "Vivado HLS " + profile["ReportVersion"]["Version"]
    res["Product family"] = user_assignment["ProductFamily"]
    res["Target device"] = user_assignment["Part"]
    clock_unit = user_assignment["unit"]
    res["Top Model Name"] = user_assignment["TopModelName"]
    res["Target CP"] = user_assignment["TargetClockPeriod"] + " " + clock_unit
    res["Estimated CP"] = perf_estimate["SummaryOfTimingAnalysis"]["EstimatedClockPeriod"] + " " + clock_unit
    res["Latency (cycles)"] = "Min {:<6}; ".format(overall_latency["Best-caseLatency"]) + \
                              "Max {:<6}".format(overall_latency["Worst-caseLatency"])
    res["Interval (cycles)"] = "Min {:<6}; ".format(overall_latency["Interval-min"]) + \
                               "Max {:<6}".format(overall_latency["Interval-max"])

    #res["HLS Version"] = names["Config"]["Name"] + profile[names["Config"]["Version"]["ReportVersion"]][names["Config"]["Version"]["Version"]]
    
    est_resources = area_estimate["Resources"]
    avail_resources = area_estimate["AvailableResources"]
    resources = {}
    for name in ["BRAM_18K", "DSP48E", "FF", "LUT"]:
    #for name in names["Resources"]["Category"]:
        item = [est_resources[name], avail_resources[name]]
        item.append("{}%".format(round(int(item[0])/int(item[1])*100)))
        resources[name] = item.copy()
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
    clock_unit = overall_latency["unit"]
    summary = perf_estimate["SummaryOfLoopLatency"]

    info_table = Displayer(clock_unit)
    #info_table.get_loops(summary)
    #info_table.get_category(summary)
    #info_table.scan_range(summary)
    #info_table.init_data(summary)
    info_table.init_table(summary)
    info_table.collect_data(summary)

    if print_flag:
        print(table)
    return info_table

def report_stats(target, folder):
    path = folder
    if target.tool.name == "vivado_hls":
        if os.path.isdir(os.path.join(path, "out.prj")):
            return parse_xml(path)
        else:
            raise RuntimeError("Not found out.prj folder")

    elif target.tool.name == "aocl":
        if os.path.isdir(os.path.join(path, "kernel/reports")):
            return parse_js(path)
    else:
        raise RuntimeError("tool {} not yet supported".format(target.tool.name))
