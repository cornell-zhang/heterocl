def tvm_to_primitive(expr):
    """Converts tvm data types into standard primitives

    Parameters
    ----------
    expr : tvm.expr or any type


    Returns
    -------
    primitive
    """
    if not isinstance(expr,int):
        return expr.value
    else:
        return expr

def update_if(cur_dict, ap_dict):
    """Adds item to the dict if key is not already in dict

    Parameters
    ----------
    cur_dict : dict
        The dictionary we wish to update

    ap_dict : dict
        The dictionary we want to append to the current dictionary
        without overwriting any current keys

    Returns
    -------
    cur_dict : dict
        The dictionary that has been updated
    """
    assert type(cur_dict) == type(ap_dict) == dict
    "type must be a dict"
    for key in ap_dict:
        if key not in cur_dict:
            cur_dict[key] = ap_dict[key]
    return cur_dict

#move to util
def partial_flatten(l):
    """Flattens first layer of lists
    i.e.: [1,[2],[[3]]] -> [1,2,[3]]

    Parameters
    ----------
    l : list
        the list we wish to partially flatten

    Returns
    -------
    _list : list
        the list that has been partially flattened

    """
    _list = []
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                _list.append(item)
        else:
            _list.append(sublist)
    return _list

#move to util
def full_flatten(l):
    """Fully flattens the list (excluding str and bytes)
    i.e.: [1,[2],[[3]]] -> [1,2,3]

    Parameters
    ----------
    l : list
        the list we wish to fully flatten

    Returns
    -------
    _ : list
        the list that was fully flattened
    """
    def _flatten(l):
        for x in l:
            if isinstance(
                    x, (list, tuple)) and not isinstance(
                    x, (str, bytes)):
                for item in _flatten(x):
                    yield item
            else:
                yield x
    return list(_flatten(l))

#move to util
def fst(l):
    """Returns the first item in any list

    Parameters
    ---------
    l : list
        the list we want to extract the first item from

    Returns
    -------
        first item in list
    """
    if isinstance(l, list):
        return fst(l[0])
    else:
        return l

def isPureList(item):
    """determines if a list is a list and not a sequence of chars or bytes

    Parameters
    ---------
    item: list
        object the user is trying to determine is a pure list

    Returns
    -------
        if the list meets the criteria stated above
    """
    return isinstance(item, list) and not isinstance(item, (str, bytes))

def gen_tpl(var, env):
    def change_list(var, env):
        l=[]
        for inx in range(len(var)):
            if(not isPureList(var[inx])):
                l.append(env[var[inx]])
            else:
                l.append(change_list(var[inx],env))
        return tuple(l)
    return change_list(var,env)