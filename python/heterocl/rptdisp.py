import pandas as pd
from tabulate import tabulate

class RptDisp(object):
  """
  A class used for the report table that could be queried.

  Attributes
  ----------
  _category : list

  _category_aux : list

  _loop_name : list

  _max_level : int

  _data : dict

  unit : str
    Unit of each information

  Methods
  ----------
  __is_range(v, key)

  scan_range(obj)

  __get_value(v, key, minmax)

  __info_extract(obj, key, minmax, col)

  init_data(obj)
  
  get_loops(obj)
  
  __select_loops(loops)
  
  __select_levels(loops, level)
  
  __select_cols(cols)
  
  display(loops=None, level=None, cols=None)

  """

  def __init__(self, unit):
    """
    Parameters
    ----------
    None
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
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
    boolean   
    """
    return isinstance(v.get(key), dict)

  def scan_range(self, obj):
    """
    Scans the parsed xml file to check which categories have range 
    values and updates _category_aux accordingly. Also, it initializes
    _data to be used in displaying the report data.

    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
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

  def __get_value( self, v, key, minmax ):
    """
    Gets the value associated with _key_. If the value is a range
    value, get the appropriate 'min' or 'max' value, determined by
    _minmax_.

    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
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
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
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
    
    Returns
    ----------

    Return Type
    ----------
      
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
    Initializes the loop
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
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

  # Given the list of loops that the user is interested in, select only those rows
  def __select_loops(self, loops):
    """
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
    """
    selected = []
    for l in loops:
      for k in self._loop_name_aux:
        if (l in k):
          selected.append(k)
    return selected
  
  # Select only the loops that are within the range of level                    
  def __select_levels(self, loops, level):
    """
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
    """
    rows = []
    for k in loops:
      lev = k.count('+')
      if lev <= level:
        rows.append(k)
    return rows 

  # TODO: Figure out exact pattern matching
  def __select_cols(self, cols):
    """
    Parameters
    ----------
    
    Returns
    ----------

    Return Type
    ----------
      
    """
    ncols = []
    for c in cols:
      for ca in self._category_aux:
        if (c in ca):
          ncols.append(ca)
          print(ncols)
    return ncols


  # loops - list of stage names
  # level - integer
  # cols  - list of column categories
  #         TODO: one could either provide the default one of the five or the exact 
  #               _select_cols buggy currently
  def display(self, loops=None, level=None, cols=None):
    """
    Display the report file.

    Parameters
    ----------
    loops
    
    level

    cols
 
    Returns
    ----------

    Return Type
    ----------
      
    """
    if loops is None:
      loops = self._loop_name_aux
    if level is None:
      level = self._max_level
    if cols is None:
      cols = self._category_aux

    selected = self.__select_loops(loops)
    rows = self.__select_levels(selected, level)
    #ncols = self.__select_cols(cols)
    alignment = ('left',)
    for i in range(len(cols)):
      alignment = alignment + ('right',)

    df = pd.DataFrame(data=self._data, index=self._loop_name_aux)
    headers = self._category_aux
    print(tabulate(df.loc[rows, cols], headers=headers, tablefmt='psql', colalign=alignment))
    print('* Units in '.format(self.unit))

