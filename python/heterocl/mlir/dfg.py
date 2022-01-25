class DFGNode(object):
    def __init__(self, tensor):
        self.name = tensor.name
        self.tensor = tensor
        self.device = None
        self.children = []
        self.parents = []

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)

    def has_children(self):
        if len(self.children) == 0:
            return False
        else:
            return True

    def set_device(self, device):
        self.device = device


class DataflowGraph(object):
    def __init__(self, name="", inputs=[]):
        self.name = name
        self.roots = []
        self.node_map = {}
        for tensor in inputs:
            self.roots.append(self.create_node(tensor))

    def create_node(self, tensor):
        name = tensor.name
        if name in self.node_map:
            node = self.node_map[name]
        else:
            node = DFGNode(tensor)
            self.node_map[name] = node
        return node

    def set_leaves(self, outputs):
        self.leaves = []
        for output in outputs:
            if output.name not in self.node_map:
                raise RuntimeError("Output not in DFG node map")
            elif self.node_map[output.name].has_children():
                raise RuntimeError("Output is not leaf")
            self.leaves.append(self.node_map[output.name])

    def add_edge(self, src, dst):
        src_node = self.create_node(src)
        dst_node = self.create_node(dst)
        src_node.add_child(dst_node)
        dst_node.add_parent(src_node)

    def add_edges(self, src_nodes, dst_nodes):
        if not isinstance(src_nodes, list):
            src_nodes = [src_nodes]
        if not isinstance(dst_nodes, list):
            dst_nodes = [dst_nodes]
        for src in src_nodes:
            for dst in dst_nodes:
                self.add_edge(src, dst)

    def propagate_annotation(self, tensor, attr):
        name = tensor.name
        node = self.node_map[name]
        node.set_device(attr)
        def set_annotation(src, dst):
            dst.set_device(attr)
        self._dfs(node, set_annotation)

    def visit(self, func):
        for node in self.roots:
            self._dfs(node, func)

    def _dfs(self, node, func=None):
        for child in node.children:
            func(node, child)
            self._dfs(child, func)

    def dump(self):
        print("Dataflow graph:")

        def print_node(src, dst):
            print(src.name, "->", dst.name)
        self.visit(print_node)

    def visualize(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        # from networkx.drawing.nx_agraph import write_dot, graphviz_layout

        edges = []

        def append_edge(src, dst):
            edges.append((src.name, dst.name))
        self.visit(append_edge)

        graph_name = "dfg_{}".format(self.name)
        nx_G = nx.from_edgelist(edges, create_using=nx.DiGraph)
        # write_dot(nx_G,'{}.dot'.format(graph_name))
        # pos = graphviz_layout(nx_G)
        # nx.draw_networkx(nx_G, pos)
        color_map = []
        for node in nx_G:
            if self.node_map[node].device == None:
                color_map.append("blue")
            elif self.node_map[node].device in ["host", "CPU"]:
                color_map.append("green")
            elif self.node_map[node].device in ["device", "FPGA"]:
                color_map.append("red")
            else:
                print(node, self.node_map[node].device)
                raise RuntimeError("Incorrect devices")
        nx.draw_networkx(nx_G, node_color=color_map)
        for color, device in [("blue", "None"), ("green", "CPU"), ("red", "FPGA")]:
            plt.scatter([],[], c=color, label=device)
        plt.legend()
        plt.savefig("{}.png".format(graph_name), format="png", dpi=200)

    def graph_partition(self):
        # first check if the requested data placement is valid
        flag = True
        def check_valid(src, dst):
            if dst.device == None:
                flag = False
        self.visit(check_valid)
        if not flag:
            self.visualize()
            raise RuntimeError("There exists DFG nodes not labeled target devices")
        # need to duplicate the boundary node that cuts across host & device
        visited = []
        def duplicate(src, dst):
            if src in visited:
                return
            else:
                visited.append(src)
            if src.device in ["device", "FPGA"] and dst.device in ["host", "CPU"]:
                dst_device = DFGNode(dst.tensor)
                dst_device.name = dst.name + "_d"
                self.node_map[dst_device.name] = dst_device
                # new node's children and parents
                dst_device.add_child(dst)
                dst_device.device = "FPGA"
                dst_device.parents = dst.parents
                # dst's parents
                dst.parents = [dst_device]
                # src's children
                for parent in dst_device.parents:
                    parent.children.remove(dst)
                    parent.children.append(dst_device)
            elif src.device in ["host", "CPU"] and dst.device in ["device", "FPGA"]:
                dst_host = DFGNode(dst.tensor)
                dst_host.name = dst.name + "_h"
                self.node_map[dst_host.name] = dst_host
                # new node's children and parents
                dst_host.add_child(dst)
                dst_host.device = "CPU"
                dst_host.parents = dst.parents
                # dst's parents
                dst.parents = [dst_host]
                # src's children
                for parent in dst_host.parents:
                    parent.children.remove(dst)
                    parent.children.append(dst_host)
            elif src in self.roots and src.device in ["device", "FPGA"]:
                src_host = DFGNode(src.tensor)
                src_host.name = src.name + "_h"
                self.node_map[src_host.name] = src_host
                # new node's children and parents
                src_host.add_child(src)
                src_host.device = "CPU"
                src_host.parents = []
                # dst's parents
                src.parents = [src_host]
                # src's children
                self.roots.remove(src)
                self.roots.append(src_host)
            else:
                pass
        self.visit(duplicate)
        self.visualize()