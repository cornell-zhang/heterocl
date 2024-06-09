import heterocl as hcl
import numpy as np
import sys
import hcl_mlir

from hcl_mlir import Module, Context, Location
from hcl_mlir.dialects import hcl as hcl_d

# Utility Function
def ProfileKernel(module):
    '''
    Count # of Memory Access of the kernel. The input `s` is schedule. (s = hcl.create_schedule(...))
    '''
    
    def FindOutputMemref(module):
        '''
        Find the output memref of the kernel.
        '''
        for func in module.body.operations:
            for i, op in enumerate(func.entry_block.operations):
                if isinstance(op, hcl_mlir.dialects.std.ReturnOp):
                    if len(op.operands) == 0:
                        return None
                    elif len(op.operands) != 1:
                        raise Exception(f"Error: Return Op has # of operand: {len(op.operands)}")
                    output_arg = op.operands[0]
        return output_arg

    # print("========= Module Begin =========")
    # print(module)
    # print("========= Module End =========\n")

    output_arg = FindOutputMemref(module)
    
    # Count Memory Access
    for func in module.body.operations:
        num_memory_access = 0
        num_arith_op = 0
        # print(f"func.entry_block.arguments: {func.entry_block.arguments}")
        module_input_args = func.entry_block.arguments # list of function arguments, including MemRef Value. 


        # Packaging all the args into module_args
        module_args = (module_input_args, output_arg)

        for i, op in enumerate(func.entry_block.operations):
            induct_var_dict = dict()

            # check if op contains AffineLoadOp, print it out when found. If this is a AffineForOp, recursively check the op.body.operations (also record the loop bound). 
            num_memory_access, num_arith_op = CountMemoryAccess(op, num_memory_access, num_arith_op, module_args, induct_var_dict)
            # print(f"# of Memory Access: {num_memory_access}")
    return num_memory_access, num_arith_op

# Utility Function use by ProfileKernel(s)
def match_memref(module_args, memref_name):
    '''
    Find if the memref_name is in the func_args. Used in LoadOp and StoreOp. 
    If True, this is off-chip memory access; if False, this is on-chip memory access. 
    '''

    module_input_args, output_arg = module_args

    for arg in module_input_args:
        if (memref_name == arg or memref_name == output_arg):
            return True
    return False


def get_loopbound(op):
    '''
    trip count of the loop of the op
    '''
    if isinstance(op, hcl_mlir.dialects.affine.AffineForOp):
        induct_var = op.induction_variable
       
        for attr in op.attributes:
            # print(f"attr: {attr}")
            if (attr.name == 'lower_bound'):
                lower_bound = int(str(attr.attr).split(")>")[0].split("-> (")[1])
            if (attr.name == 'upper_bound'):
                upper_bound = int(str(attr.attr).split(")>")[0].split("-> (")[1])
            if (attr.name == 'step'):
                step = int(str(attr.attr).split(" : ")[0])
        trip_count = (upper_bound - lower_bound) // step

        loop_var_dict = {op.induction_variable: trip_count}

        return induct_var, trip_count
    else:
        print("Error: not an AffineForOp")
        sys.exit()


def CountMemoryAccess(op, memory_access, arith_op, module_args, induct_var_dict):
    '''
    Count # of Memory Access. When we found a Load or Store op, we add the trip_count under this loop to the Memory Access. 
    '''
    def get_tripcount(induct_var_dict):
        tp = 1
        for value in induct_var_dict.values():
            tp *= value
        return tp

    def get_cond_reduction(cond):
        '''
        Get reduction size for the induction variable, based on AffineIfOp's condition.
        String matching here.  
        e.g. 
            cond = affine_set<(d0) : (d0 - 2 >= 0)>
            cond_reduct = 2
            return cond_reduct
        '''
        cond_reduct = int(str(cond).split(" >= 0")[0].split("- ")[-1]) # TODO: only works for above #set pattern. 
        return cond_reduct
        

    if isinstance(op, (hcl_mlir.dialects.affine.AffineLoadOp, hcl_mlir.dialects.affine.AffineStoreOp)):
        tp = get_tripcount(induct_var_dict)
        # print(f"Found Load / Store Op: {op}, TripCount: {tp}")
        match_result = match_memref(module_args, op.memref)
        
        if match_result: # off-chip memory access
            memory_access += tp
            # print(f"Off-Chip Memory Access")
        else:
            pass
            # print(f"On-Chip Memory Access")

    elif isinstance(op, (hcl_mlir.dialects.arith.AddIOp, hcl_mlir.dialects.arith.MulIOp, hcl_mlir.dialects.arith.AddFOp, hcl_mlir.dialects.arith.MulFOp)):
        tp = get_tripcount(induct_var_dict)
        # print(f"Found Arith Op: {op}, TripCount: {tp}")
        arith_op += tp


    elif isinstance(op, hcl_mlir.dialects.affine.AffineForOp):
        # print(f"Found AffineForOp")
        induct_var, tp = get_loopbound(op)
        induct_var_dict[induct_var] = tp

        for inner_op in op.body.operations:
            memory_access, arith_op = CountMemoryAccess(inner_op, memory_access, arith_op, module_args, induct_var_dict) # recursively calling this Function
        
        induct_var_dict.pop(induct_var)
        # print(f"==== AffineForOp End ====")

    elif isinstance(op, hcl_mlir.dialects.affine.AffineIfOp):
        # print(f"Found AffineIfOp")
        cond_reduct_size = get_cond_reduction(op.attributes[0].attr)

        induct_var = op.operands[0]
        induct_var_dict[induct_var] -= cond_reduct_size

        for inner_op in op.then_block.operations:
            memory_access, arith_op = CountMemoryAccess(inner_op, memory_access, arith_op, module_args, induct_var_dict)

        # print("==== AffineIfOp End ====")
        induct_var_dict[induct_var] += cond_reduct_size

    else:
        pass
        # print(f"Found unknown op: {op}")
    
    # print(f"MemoryAccess: {memory_access}")
    # print(f"ArithOp: {arith_op}")
    return memory_access, arith_op


# Context
class GlobalContext(object):
    def __init__(self):
        self.ctx = None
        self.loc = None

    def get_context(self):
        return self.ctx

    def set_context(self):
        self.ctx = Context()
        hcl_d.register_dialect(self.ctx)
        self.loc = Location.unknown(self.ctx)

    def get_location(self):
        return self.loc

global_ctx = GlobalContext()
get_context = global_ctx.get_context
set_context = global_ctx.set_context
get_location = global_ctx.get_location



def load_mlir(filename):
    # f = open(filename, 'r')
    # print(f.read())

    set_context()

    with open(filename, 'r') as f:
        with get_context() as ctx, get_location():
            module = Module.parse(f.read(), ctx) # s.device_module
            # print(module)
            # print(module.body.operations)
            # for func in module.body.operations:
            #     print(func)
            return module


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python profile.py <filename>")
        exit(1)
    else:
        filename = sys.argv[1]

    module = load_mlir(filename)
    num_memory_access, num_arith_op = ProfileKernel(module)

    print("================ Finish Analysis ==================")
    print(f"# of Memory Access: {num_memory_access}")
    print(f"# of Arith Op: {num_arith_op}")