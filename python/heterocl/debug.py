"""Exception handler in HeteroCL

This module defines various HeteroCL exceptions. Developers are free to
add new types of exception.
"""
#pylint: disable=superfluous-parens
import sys
import traceback

class HCLError(Exception):
    """HeteroCL related exception

    User can specify additional class for the exception.

    Parameters
    ----------
    msg : str
        The error message.

    info : str, optional
        Additional class specification for the exception.
    """
    def __init__(self, msg, info=None):
        if info is not None:
            msg = info + msg
        Exception.__init__(self, msg)

class DTypeError(HCLError):
    """A subclass for specifying data type related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[Data Type]\33[0m ")

class APIError(HCLError):
    """A subclass for specifying API related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[API]\33[0m ")

class DSLError(HCLError):
    """A subclass for specifying imperative DSL related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[Imperative]\33[0m ")

class TensorError(HCLError):
    """A subclass for specifying tensor related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[Tensor]\33[0m ")

class DeviceError(HCLError):
    """A subclass for specifying device related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[Device]\33[0m ")

class AssertError(HCLError):
    """A subclass for specifying assert related exception"""
    def __init__(self, msg):
        HCLError.__init__(self, msg, "\33[1;31m[Assert]\33[0m ")

def hcl_excepthook(etype, value, tb):
    """Customized excepthook

    If the exception is a HeteroCL exception, only the traceback that
    related to user's program will be listed. All HeteroCL internal
    traceback will be hidden.
    """
    if issubclass(etype, HCLError):
        extracted_tb = traceback.extract_tb(tb)
        frame_stack = []
        for e_tb in extracted_tb:
            if '/python/heterocl/' in e_tb[0]:
                continue
            frame_stack.append(e_tb)
        for frame in traceback.format_list(frame_stack):
            sys.stdout.write(frame)
            sys.stdout.flush()
        print("\33[1;34m[HeteroCL Error]\33[0m" + value.args[0])
    else:
        sys.__excepthook__(etype, value, tb)
