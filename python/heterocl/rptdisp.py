import pandas as pd
from tabulate import tabulate

class RptDisp(object):
  """
  A class used for the report table that could be queried.

  Attributes
  ----------
  * _category (list)
    List of default latency information category.

  * _category_aux (list)
    List of latency information category with range indicators.

  * _loop_name (list)
    List of loop names without loop nest indicators.

  * _loop_name_aux (list)
    List of loop names with loop nest indicators.

  * _max_level (int)
    Maximum level of loop nest in the report file.

  * _data (dict)
    Dictionary containing latency data. 

  * unit (str)
    Unit of each information.

  Methods
  ----------
  * __is_range(v, key)
    Check whether the value is a range value or not.

  * scan_range(obj)
    Scan the entire report file to see which latency category contains
    range values.

  * __get_value(v, key, minmax)
    Get the value associated with the input key.

  * __info_extract(obj, key, minmax, col)
    Extract out all the latency information from the report. 

  * init_data(obj)
    Initialize the data given the report file.  

  * get_loops(obj)
    Acquire loop names with and without loop nest indicators.
  
  * __select_loops(loops)
    Select only the loops specified by the user.
  
  * __select_levels(loops, level)
    Select only the loops that are included in the level specified by
    the user.
  
  * __select_cols(cols)
    Select only the columns specified by the user.
  
  * display(loops=None, level=None, cols=None)
    Display the report table with appropriate query arguments.
  """

  def __init__(self, unit):
    """
    Parameters
    ----------
    * unit (str) - Unit for all numerical values in the table.
    """
    self._category = ['Trip Count', 'Latency', 'Iteration Latency', 
                       'Pipeline II', 'Pipeline Depth']
    self._category_aux = []
    self._loop_name = []
    self._loop_name_aux = []
    self._max_level = 0
    self._data = {}
    self.unit = unit

  def __is_range(self, v, key):    
    """
    Check whether the value is a range value or not.

    Parameters
    ----------
    * v   (dict) - Dictionary containing all latency information
                   for a particular loop.
    * key (str)  - Latency category.

    Returns
    ----------
    True if the value is a range value. Otherwise, False. 

    Return Type
    ----------
    bool 
    """
    return isinstance(v.get(key), dict)

  def scan_range(self, obj):
    """
    Scans the parsed xml file to check which categories have range 
    values and updates _category_aux accordingly. Also, it initializes
    _data to be used in displaying the report data.

    Parameters
    ----------
    * obj (dict) - Dictionary representation of the report file.

    Returns
    ----------
    None

    Return Type
    ----------
    None  
    """
    detect_minmax = []
    for item in self._category:
      cat = item.replace(' ', '')
      tf = False
      for k, v in obj.items():
        tf = tf or self.__is_range(v, cat)
        in_k, in_v = list(v.items())[-1]
        while not isinstance(in_v, str):
          tf = tf or self.__is_range(v, cat)
          in_k, in_v = list(in_v.items())[-1]

      if tf:
        detect_minmax.append('Min ' + item)
        detect_minmax.append('Max ' + item)
      else:
        detect_minmax.append(item)

    self._category_aux = detect_minmax
    for c in self._category_aux:
      self._data[c] = []

  def __get_value(self, v, key, minmax):
    """
    Gets the value associated with _key_. If the value is a range
    value, get the appropriate 'min' or 'max' value, determined by
    _minmax_.

    Parameters
    ----------
    * v      (dict) - Dictionary containing all latency information
                      for a particular loop.
    * key    (str)  - Latency category.
    * minmax (str)  - Range indicator (min or max).
 
    Returns
    ----------
    Latency value of the loop with category 'key'. 

    Return Type
    ----------
    str
    """
    num = v.get( key )
    if isinstance( num, str ):
      val = str( num )
    elif isinstance( num, dict ):
      val = num['range'][minmax]
    else:
      val = 'N/A'
    return val

  def __info_extract(self, obj, key, minmax, col):
    """
    Extract out all the latency information from the report.

    Parameters
    ----------
    * obj    (dict) - Dictionary representation of the report file.
    * key    (str)  - Latency category. 
    * minmax (str)  - Range indicator (min or max). 
    * col    (list) - Column name in the data.
    
    Returns
    ----------
    None

    Return Type
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

  def init_data(self, obj):
    """
    Initialize the _data attribute.
 
    Parameters
    ----------
    * obj (dict) - Dictionary representation of the report file. 
    
    Returns
    ----------
    None

    Return Type
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

  def get_loops(self, obj):
    """
    Initializes the loop name lists.

    Parameters
    ----------
    * obj (dict) - Dictionary representation of the report file. 

    Returns
    ----------
    None

    Return Type
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

  def __select_loops(self, loops):
    """
    Select only the loops specified by the user.

    Parameters
    ----------
    * loops (list) - List of loop names.
    
    Returns
    ----------
    List of specific pattern-matched loop names. 

    Return Type
    ----------
    list  
    """
    selected = []
    for l in loops:
      for k in self._loop_name_aux:
        if (l in k):
          selected.append(k)
    return selected
  
  def __select_levels(self, loops, level):
    """
    Select only the loops that are within the range of level.

    Parameters
    ----------
    * loops (list) - List of specified loop names.
    * level (int)  - Number indicating the maximum loop nest level to
                     print.

    Returns
    ----------
    List of loops that are within the range of level.

    Return Type
    ----------
    list
    """
    rows = []
    for k in loops:
      lev = k.count('+')
      if lev <= level:
        rows.append(k)
    return rows 

  def __select_cols(self, cols):
    """
    TODO: Exact pattern matching

    Parameters
    ----------
    * cols (list) - List of specific column names.
    
    Returns
    ----------
    List of pattern-matched column names.

    Return Type
    ----------
    list
    """
    ncols = []
    for c in cols:
      for ca in self._category_aux:
        if (c in ca):
          ncols.append(ca)
    return ncols
 
  def display(self, loops=None, level=None, cols=None):
    """
    Display the report file.

    Parameters
    ----------
    * loops (list, optional) - List of loop names (e.g., ['A', 'Y'])
    * level (int,  optional) - Maximum level of loop nest to print
    * cols  (list, optional) - List of column names (e.g., ['Trip Count'])
 
    Returns
    ----------
    None

    Return Type
    ----------
    None  
    """
    if loops is None:
      loops = self._loop_name_aux
    if level is None:
      level = self._max_level
    if cols is None:
      cols = self._category_aux

    selected = self.__select_loops(loops)
    rows = self.__select_levels(selected, level)
    # TODO: one could either provide the default one of the five or the exact 
    #       _select_cols buggy currently
    #ncols = self.__select_cols(cols)
    alignment = ('left',)
    for i in range(len(cols)):
      alignment = alignment + ('right',)

    df = pd.DataFrame(data=self._data, index=self._loop_name_aux)
    print(tabulate(df.loc[rows, cols], headers=cols, tablefmt='psql', colalign=alignment))
    print('* Units in '.format(self.unit))

