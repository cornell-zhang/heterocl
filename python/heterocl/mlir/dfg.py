from platform import node


class DFGNode(object):
    def __init__(self, tensor):
        self.name = tensor.name
        self.tensor = tensor
        self.children = []
        self.parents = []

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)


class DataflowGraph(object):
    def __init__(self, name="", inputs=[]):
        self.name = name
        self.roots = []
        self.node_map = {}
        for tensor in inputs:
            self.roots.append(self.create_node(tensor))

    def create_node(self, tensor):
        name = tensor.name.replace("_ret", "")
        if name in self.node_map:
            node = self.node_map[name]
        else:
            node = DFGNode(tensor)
            self.node_map[name] = node
        return node

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
        nx.draw_networkx(nx_G)
        plt.savefig("{}.png".format(graph_name),format="png",dpi=200)