from .tvm import expr as _expr
from .tvm import stmt as _stmt
from .tvm import make as _make
from .tvm.api import convert

class Mutator(object):

    def mutate(self, node):
        if isinstance(node, _expr.Expr):
            if isinstance(node, _expr.ConstExpr):
                return self.mutate_ConstExpr(node)
            elif isinstance(node, _expr.BinaryOpExpr):
                if isinstance(node, _expr.Add):
                    return self.mutate_Add(node)
                elif isinstance(node, _expr.Sub):
                    return self.mutate_Sub(node)
                elif isinstance(node, _expr.Mul):
                    return self.mutate_Mul(node)
                elif isinstance(node, _expr.Div):
                    return self.mutate_Div(node)
                elif isinstance(node, _expr.Mod):
                    return self.mutate_Mod(node)
                elif isinstance(node, _expr.Min):
                    return self.mutate_Min(node)
                elif isinstance(node, _expr.Max):
                    return self.mutate_Max(node)
                else:
                    return node
            elif isinstance(node, _expr.CmpExpr):
                if isinstance(node, _expr.EQ):
                    return self.mutate_EQ(node)
                elif isinstance(node, _expr.NE):
                    return self.mutate_NE(node)
                elif isinstance(node, _expr.LT):
                    return self.mutate_LT(node)
                elif isinstance(node, _expr.LE):
                    return self.mutate_LE(node)
                elif isinstance(node, _expr.GT):
                    return self.mutate_GT(node)
                elif isinstance(node, _expr.GE):
                    return self.mutate_GE(node)
                else:
                    return node
            elif isinstance(node, _expr.LogicalExpr):
                if isinstance(node, _expr.And):
                    return self.mutate_And(node)
                elif isinstance(node, _expr.Or):
                    return self.mutate_Or(node)
                elif isinstance(node, _expr.Not):
                    return self.mutate_Not(node)
                else:
                    return node
            else:
                if isinstance(node, _expr.Var):
                    return self.mutate_Var(node)
                elif isinstance(node, _expr.Cast):
                    return self.mutate_Cast(node)
                elif isinstance(node, _expr.Select):
                    return self.mutate_Select(node)
                elif isinstance(node, _expr.Load):
                    return self.mutate_Load(node)
                elif isinstance(node, _expr.Ramp):
                    return self.mutate_Ramp(node)
                elif isinstance(node, _expr.Broadcast):
                    return self.mutate_Broadcast(node)
                elif isinstance(node, _expr.Call):
                    return self.mutate_Call(node)
                elif isinstance(node, _expr.Let):
                    return self.mutate_Let(node)
                elif isinstance(node, _expr.GetBit):
                    return self.mutate_GetBit(node)
                elif isinstance(node, _expr.GetSlice):
                    return self.mutate_GetSlice(node)
                elif isinstance(node, _expr.SetBit):
                    return self.mutate_SetBit(node)
                elif isinstance(node, _expr.SetSlice):
                    return self.mutate_SetSlice(node)
                elif isinstance(node, _expr.KernelExpr):
                    return self.mutate_KernelExpr(node)
                elif isinstance(node, _expr.StreamExpr):
                    return self.mutate_StreamExpr(node)
                else:
                    return node
        elif isinstance(node, _stmt.Stmt):
            if isinstance(node, _stmt.LetStmt):
                return self.mutate_LetStmt(node)
            elif isinstance(node, _stmt.AssertStmt):
                return self.mutate_AssertStmt(node)
            elif isinstance(node, _stmt.ProducerConsumer):
                return self.mutate_ProducerConsumer(node)
            elif isinstance(node, _stmt.ExternModule):
                return self.mutate_ExternModule(node)
            elif isinstance(node, _stmt.For):
                return self.mutate_For(node)
            elif isinstance(node, _stmt.Store):
                return self.mutate_Store(node)
            elif isinstance(node, _stmt.Allocate):
                return self.mutate_Allocate(node)
            elif isinstance(node, _stmt.AttrStmt):
                return self.mutate_AttrStmt(node)
            elif isinstance(node, _stmt.Free):
                return self.mutate_Free(node)
            elif isinstance(node, _stmt.Block):
                return self.mutate_Block(node)
            elif isinstance(node, _stmt.IfThenElse):
                return self.mutate_IfThenElse(node)
            elif isinstance(node, _stmt.Evaluate):
                return self.mutate_Evaluate(node)
            elif isinstance(node, _stmt.KernelDef):
                return self.mutate_KernelDef(node)
            elif isinstance(node, _stmt.KernelStmt):
                return self.mutate_KernelStmt(node)
            elif isinstance(node, _stmt.Return):
                return self.mutate_Return(node)
            elif isinstance(node, _stmt.Break):
                return self.mutate_Break(node)
            elif isinstance(node, _stmt.While):
                return self.mutate_While(node)
            elif isinstance(node, _stmt.StreamStmt):
                return self.mutate_StreamStmt(node)
            else:
                return node
        elif isinstance(node, tuple):
            return self.mutate_Tuple(node)
        elif isinstance(node, list):
            return self.mutate_List(node)
        elif callable(node):
            return self.mutate_Function(node)
        else:
            return node

    def mutate_ConstExpr(self, node):
        return node

    def mutate_BinOp(self, binop, node):
        a = self.mutate(node.a)
        b = self.mutate(node.b)
        return binop(a, b)

    def mutate_Add(self, node):
        return self.mutate_BinOp(_make.Add, node)

    def mutate_Sub(self, node):
        return self.mutate_BinOp(_make.Sub, node)

    def mutate_Mul(self, node):
        return self.mutate_BinOp(_make.Mul, node)

    def mutate_Div(self, node):
        return self.mutate_BinOp(_make.Div, node)

    def mutate_Mod(self, node):
        return self.mutate_BinOp(_make.Mod, node)

    def mutate_Min(self, node):
        return self.mutate_BinOp(_make.Min, node)

    def mutate_Max(self, node):
        return self.mutate_BinOp(_make.Max, node)

    def mutate_EQ(self, node):
        return self.mutate_BinOp(_make.EQ, node)

    def mutate_NE(self, node):
        return self.mutate_BinOp(_make.NE, node)

    def mutate_LT(self, node):
        return self.mutate_BinOp(_make.LT, node)

    def mutate_LE(self, node):
        return self.mutate_BinOp(_make.LE, node)

    def mutate_GT(self, node):
        return self.mutate_BinOp(_make.GT, node)

    def mutate_GE(self, node):
        return self.mutate_BinOp(_make.GE, node)

    def mutate_And(self, node):
        return self.mutate_BinOp(_make.And, node)

    def mutate_Or(self, node):
        return self.mutate_BinOp(_make.Or, node)

    def mutate_Not(self, node):
        a = self.mutate(node.a)
        return _make.Not(a)

    def mutate_Var(self, node):
        return node

    def mutate_Cast(self, node):
        value = self.mutate(node.value)
        return _make.Cast(node.dtype, value)

    def mutate_Select(self, node):
        condition = _make.Cast("uint1", self.mutate(node.condition))
        true_value = convert(self.mutate(node.true_value))
        false_value = convert(self.mutate(node.false_value))
        return _make.Select(condition, true_value, _make.Cast(true_value.dtype, false_value))

    def mutate_Load(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        predicate = self.mutate(node.predicate)
        return _make.Load(node.dtype, buffer_var, index, predicate)

    def mutate_Ramp(self, node):
        base = self.mutate(node.base)
        stride = self.mutate(node.stride)
        return _make.Ramp(base, stride, node.lanes)

    def mutate_Broadcast(self, node):
        value = self.mutate(node.value)
        return _make.Broadcast(value, node.lanes)

    def mutate_Call(self, node):
        args = []
        for arg in node.args:
            args.append(self.mutate(arg))
        return _make.Call(node.dtype, node.name, args, node.call_type, node.func, node.value_index)

    def mutate_Let(self, node):
        var = self.mutate(node.var)
        value = self.mutate(node.value)
        body = self.mutate(node.body)
        return _make.Let(var, value, body)

    def mutate_GetBit(self, node):
        a = self.mutate(node.a)
        index = self.mutate(node.index)
        return _make.GetBit(a, index)

    def mutate_GetSlice(self, node):
        a = self.mutate(node.a)
        index_left = self.mutate(node.index_left)
        index_right = self.mutate(node.index_right)
        return _make.GetSlice(a, index_left, index_right)

    def mutate_SetBit(self, node):
        a = self.mutate(node.a)
        value = self.mutate(node.value)
        index = self.mutate(node.index)
        return _make.SetBit(a, value, index)

    def mutate_SetSlice(self, node):
        a = self.mutate(node.a)
        value = self.mutate(node.value)
        index_left = self.mutate(node.index_left)
        index_right = self.mutate(node.index_right)
        return _make.SetSlice(a, value, index_left, index_right)

    def mutate_KernelExpr(self, node):
        args = self.mutate(node.args)
        return _make.KernelExpr(node.dtype, args, node.name)

    def mutate_StreamExpr(self, node):
        args = self.mutate(node.args)
        return _make.StreamExpr(node.dtype, args, node.name)

    # statements
    def mutate_LetStmt(self, node):
        var = self.mutate(node.var)
        value = self.mutate(node.value)
        body = self.mutate(node.body)
        return _make.LetStmt(var, value, body)

    def mutate_AssertStmt(self, node):
        condition = self.mutate(node.condition)
        message = self.mutate(node.message)
        body = self.mutate(node.body)
        return _make.AssertStmt(condition, message, body)

    def mutate_ProducerConsumer(self, node):
        body = self.mutate(node.body)
        return _make.ProducerConsumer(node.func, node.is_producer, body)
        
    def mutate_ExternModule(self, node):
        body = self.mutate(node.body)
        return _make.ExternModule(node.attr_key, node.value, body,
                node.annotate_keys, node.annotate_values)

    def mutate_For(self, node):
        loop_var = self.mutate(node.loop_var)
        _min = self.mutate(node.min)
        extent = self.mutate(node.extent)
        body = self.mutate(node.body)
        return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)

    def mutate_Store(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        value = self.mutate(node.value)
        predicate = self.mutate(node.predicate)
        return _make.Store(buffer_var, value, index, predicate)

    def mutate_Allocate(self, node):
        buffer_var = self.mutate(node.buffer_var)
        extents = self.mutate(node.extents)
        condition = self.mutate(node.condition)
        body = self.mutate(node.body)
        return _make.Allocate(buffer_var, node.dtype, extents, condition, body)

    def mutate_AttrStmt(self, node):
        value = self.mutate(node.value)
        body = self.mutate(node.body)
        return _make.AttrStmt(node.node, node.attr_key, value, body)

    def mutate_Free(self, node):
        buffer_var = self.mutate(node.buffer_var)
        return _make.Free(buffer_var)

    def mutate_Block(self, node):
        first = self.mutate(node.first)
        rest = self.mutate(node.rest)
        return _make.Block(first, rest)

    def mutate_IfThenElse(self, node):
        condition = self.mutate(node.condition)
        then_case = self.mutate(node.then_case)
        else_case = self.mutate(node.else_case)
        return _make.IfThenElse(condition, then_case, else_case)

    def mutate_Evaluate(self, node):
        value = self.mutate(node.value)
        return _make.Evaluate(value)

    def mutate_KernelDef(self, node):
        args = self.mutate(node.args)
        body = self.mutate(node.body)
        ret_void = self.mutate(node.ret_void)
        return _make.KernelDef(args, body, ret_void, node.ret_type, node.name)

    def mutate_KernelStmt(self, node):
        args = self.mutate(node.args)
        return _make.KernelStmt(args, node.name)

    def mutate_StreamStmt(self, node):
        args = self.mutate(node.args)
        return _make.StreamStmt(node.dtype, args, node.name)

    def mutate_Return(self, node):
        value = self.mutate(node.value)
        return _make.Return(value)

    def mutate_Break(self, node):
        return _make.Break()

    def mutate_While(self, node):
        condition = self.mutate(node.condition)
        bdoy = self.mutate(node.body)
        return _make.While(condition, body)

    def mutate_Tuple(self, node):
        _list = list(node)
        _list = self.mutate(_list)
        return tuple(_list)

    def mutate_List(self, node):
        _len = len(node)
        _list = []
        for i in range(0, _len):
            _list.append(self.mutate(node[i]))
        return _list

    def mutate_Function(self, node):
        return node
