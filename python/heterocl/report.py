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

    def __checker(self, lst):
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
            if elem[0]:
                valid |= True
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
        obj = elem[0]
        
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
            frame.append((obj[il]))
                                                                 
        if len(frame) == 0:
            frame.append(({}))
                                                                 
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
    
        frame_lst = []
    
        for k in keys:
            frame_lst.append((obj[k], [], k, 0, []))
    
        correct = self.__checker(frame_lst)
        
        while correct:
            frame_lst = list(map(self.__member_init, frame_lst))
    
            frame_lst = [item for elem in frame_lst for item in elem]
    
            correct = self.__checker(frame_lst)
        
        self._max_level = max([x[3] for x in frame_lst])
    
        filtered = [x[1] for x in frame_lst]
        filtered_aux = [x[4] for x in frame_lst]
        
        self._loop_name = [item for elem in filtered for item in elem]
        self._loop_name = list(dict.fromkeys(self._loop_name))
    
        self._loop_name_aux = [item for elem in filtered_aux for item in elem]
        self._loop_name_aux = list(dict.fromkeys(self._loop_name_aux))
                                                                               
        self._category_aux = list(self._data.keys())
    
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
                                                                      
        for k in keys:
            frame_lst.append((obj[k]))
                                                                      
        correct = self.__checker(frame_lst)
        
        while correct:
            frame_lst = list(map(self.__data_acquisition, frame_lst))
                                                                      
            frame_lst = [item for elem in frame_lst for item in elem]
                                                                      
            correct = self.__checker(frame_lst)
    
    def get_max(self, col):
        """Form a tuple list that sorts loops in a decreasing order with
        respect to the latency information of the specified latency category.
                   
        Parameters
        ----------
        col: str
            Latency category name.
                   
        Returns
        ----------
        list
            Tuple list with loop names and its corresponding latency value. 
        """
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

    user_assignment = profile["UserAssignments"]
    perf_estimate = profile["PerformanceEstimates"]
    area_estimate = profile["AreaEstimates"]
    overall_latency = perf_estimate["SummaryOfOverallLatency"]

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

    
    est_resources = area_estimate["Resources"]
    avail_resources = area_estimate["AvailableResources"]
    resources = {}
    for name in ["BRAM_18K", "DSP48E", "FF", "LUT"]:
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
