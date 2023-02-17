# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=dangerous-default-value


class DFGNode:
    def __init__(self, tensor):
        self.name = tensor.name
        self.tensor = tensor
        self.device = None
        self.children = []
        self.parents = []
        self.states = []
        self.base = None

    def add_child(self, child):
        if len(self.states) != 0:
            self.states[-1].children.append(child)
        else:
            self.children.append(child)

    def add_parent(self, parent):
        if len(parent.states) != 0:
            self.parents.append(parent.states[-1])
        else:
            self.parents.append(parent)

    def add_state(self, state):
        state.base = self
        self.states.append(state)

    def has_children(self):
        return len(self.children) != 0

    def set_device(self, device):
        self.device = device


class DataflowGraph:
    def __init__(self, name="", inputs=[]):
        self.name = name
        self.roots = []
        self.leaves = []
        self.node_map = {}
        self.device_map = {}
        for tensor in inputs:
            self.roots.append(self.create_node(tensor))
        self.subgraph = {"inputs": [], "outputs": []}
        self.host_xcel_place = False

    def has_host_xcel_place(self):
        return self.host_xcel_place

    def create_node(self, tensor):
        name = tensor.name
        if name in self.node_map:
            node = self.node_map[name]
        else:
            node = DFGNode(tensor)
            self.node_map[name] = node
        return node

    def add_edge(self, src, dst, stateful=False):
        if src.name == dst.name:
            return
        src_node = self.create_node(src)
        dst_node = self.create_node(dst)
        src_node.add_child(dst_node)
        dst_node.add_parent(src_node)
        if stateful:
            src_node.add_state(dst_node)

    def add_edges(self, src_nodes, dst_nodes):
        if not isinstance(src_nodes, list):
            src_nodes = [src_nodes]
        if not isinstance(dst_nodes, list):
            dst_nodes = [dst_nodes]
        for src in src_nodes:
            for dst in dst_nodes:
                self.add_edge(src, dst)

    def visit(self, func):
        visited = set()
        for node in self.roots:
            self._dfs(node, visited, func)

    def _dfs(self, node, visited, func=None):
        if node.name in visited:
            return
        visited.add(node.name)
        for child in node.children:
            func(node, child)
            self._dfs(child, visited, func)

    def dump(self):
        print("Dataflow graph:")

        def print_node(src, dst):
            print(src.name, "->", dst.name)

        self.visit(print_node)

    def visualize(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout

        plt.figure(figsize=(8, 5), dpi=200)

        edges = []

        def append_edge(src, dst):
            edges.append((src.name, dst.name))

        self.visit(append_edge)

        graph_name = f"dfg_{self.name}"
        nx_G = nx.from_edgelist(edges, create_using=nx.DiGraph)
        write_dot(nx_G, f"{graph_name}.dot")
        pos = graphviz_layout(nx_G, prog="dot")
        color_map = []
        for node in nx_G:
            if self.node_map[node].device is None:
                color_map.append("blue")
            elif self.node_map[node].device in {"host", "CPU"}:
                color_map.append("green")
            elif self.node_map[node].device in {"device", "FPGA"}:
                color_map.append("red")
            else:
                print(node, self.node_map[node].device)
                raise RuntimeError("Incorrect devices")
        nx.draw_networkx(nx_G, pos, node_color=color_map)
        # nx.draw_networkx(nx_G, node_color=color_map)
        for color, device in (("blue", "None"), ("green", "CPU"), ("red", "FPGA")):
            plt.scatter([], [], c=color, label=device)
        plt.legend(loc=1)
        plt.savefig(f"{graph_name}.png", format="png", dpi=200)

    def propagate_annotation(self, tensor, attr):
        name = tensor.name
        node = self.node_map[name]

        # pylint: disable=unused-argument
        def set_annotation(src, dst):
            dst.set_device(attr)

        if attr == "CPU":
            node.set_device("FPGA")
        elif attr == "FPGA":
            node.set_device("CPU")
        # set next stage on device
        visited = set()
        self._dfs(node, visited, set_annotation)
        self.host_xcel_place = True

    def create_device_map(self):
        flag = True
        has_xcel = False

        def check_valid(src, dst):
            nonlocal flag, has_xcel
            self.device_map[src.name] = src.device
            self.device_map[dst.name] = dst.device
            if src.device is None or dst.device is None:
                flag = False
            if src.device not in ["CPU", None] or dst.device not in ["CPU", None]:
                has_xcel = True

        self.visit(check_valid)

        if not has_xcel:  # label all the graph nodes as CPU

            def label_cpu(src, dst):
                self.device_map[src.name] = "CPU"
                self.device_map[dst.name] = "CPU"
                src.device = "CPU"
                dst.device = "CPU"

            self.visit(label_cpu)
            flag = True
        return flag

    def graph_partition(self, show_partition=False):
        # first check if the requested data placement is valid
        for node in self.roots:
            if node.device is None:
                node.device = "CPU"
        if not self.create_device_map():
            self.visualize()
            raise RuntimeError("There exists DFG nodes not labeled target devices")

        def extract_subgraph(src, dst):
            if src.device in {"host", "CPU"} and dst.device in {"device", "FPGA"}:
                if src not in self.subgraph["inputs"]:
                    self.subgraph["inputs"].append(src)
            elif src.device in {"device", "FPGA"} and dst.device in {"host", "CPU"}:
                if src not in self.subgraph["outputs"]:
                    self.subgraph["outputs"].append(src)
            else:
                pass

        self.visit(extract_subgraph)
        for output in self.leaves:
            if output.device in {"device", "FPGA"}:
                self.subgraph["outputs"].append(output)
        if show_partition:
            self.visualize()
