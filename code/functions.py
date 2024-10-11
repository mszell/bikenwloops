def normalize_data(data):
    return list((data - np.min(data)) / (np.max(data) - np.min(data)))


def get_vertex_size_constant(N):
    """
    Calculate a constant node size for plotting given the number of
    nodes N. Large values of around N>666 map to 1, small values
    N<154 to 7, immediate values in-between.
    """
    return round(min(max(1000 / N, 1), 7))


def get_edgewidth_constant(L):
    """
    Calculate a constant link size for plotting given the number of
    links L. Large values of around L>666 map to 0.5, small values
    L<133 to 3, immediate values in-between, rounded in 0.5 steps.
    """
    return round(2 * min(max(333 / L, 0.5), 3)) / 2


def get_vertex_sizes(loopinfo, max_node_size=20):
    """
    Calculate a node size for each node in the loopinfo dict
    for plotting given the number of loops in the node. The
    largest value gets size max_node_size.
    Returns a list of node sizes (floats).
    """
    vertex_sizes = []
    for k in range(len(loopinfo.keys())):
        try:
            if PLOTLOGSCALE:
                vertex_sizes.append(np.log2(len(loopinfo[k]["loops"]) + 1.01))
            else:
                vertex_sizes.append(len(loopinfo[k]["loops"]))
        except:
            vertex_sizes.append(0)

    numloops_max = max(vertex_sizes)
    vertex_sizes = [i / (numloops_max / max_node_size) for i in vertex_sizes]
    return vertex_sizes, numloops_max


def get_layout(G, nodes_id, nodes_coords):
    named_vertex_list = G.vs()["name"]
    layout = []
    for n in named_vertex_list:
        try:
            pos = nodes_id.index(n)
        except:
            print("There was an invalid node with name: " + str(n))
        layout.append(nodes_coords[pos])
    return layout


def plot_check(G, nodes_id, nodes_coords, vertex_size=7, edge_width=2, edge_color=None):
    fig, ax = plt.subplots()
    layout = get_layout(G, nodes_id, nodes_coords)
    if edge_color is None:
        ig.plot(
            G,
            target=ax,
            vertex_size=vertex_size,
            layout=layout,
            vertex_color="green",
            edge_width=edge_width,
        )
    else:
        ig.plot(
            G,
            target=ax,
            vertex_size=vertex_size,
            layout=layout,
            edge_color=edge_color,
            vertex_color="green",
            edge_width=edge_width,
        )
    return fig


def get_loop_length(c):
    l = 0
    cl = len(c)
    for i in range(cl):
        l += Gnx[c[i % cl]][c[(i + 1) % cl]]["weight"]
    return l


def get_loop_max_slope(c):
    ms = 0
    cl = len(c)
    for i in range(cl):
        ms = max([ms, Gnx[c[i % cl]][c[(i + 1) % cl]]["max_slope"]])
    return ms


def get_loop_poi_diversity(c):
    pd = [0, 0, 0]  # facilities, services, attractions
    cl = len(c)
    for i in range(cl):
        pd = [
            pd[0] or Gnx[c[i % cl]][c[(i + 1) % cl]]["has_facility"],
            pd[1] or Gnx[c[i % cl]][c[(i + 1) % cl]]["has_service"],
            pd[2] or Gnx[c[i % cl]][c[(i + 1) % cl]]["has_attraction"],
        ]
    return sum(pd)


def get_loop_water_profile(c):
    wp = tuple()
    cl = len(c)
    l = 0
    for i in range(cl):
        l_this = Gnx[c[i % cl]][c[(i + 1) % cl]]["weight"]
        l += l_this
        if Gnx[c[i % cl]][c[(i + 1) % cl]][
            "has_water"
        ]:  # If link has water, assume it is half the link length in
            wp += (l + l_this / 2,)
            l = l_this / 2
    return wp


def mask_node(nodeloopinfo, mask):
    return {
        "loops": list(compress(nodeloopinfo["loops"], mask)),
        "lengths": list(compress(nodeloopinfo["lengths"], mask)),
        "numnodes": list(compress(nodeloopinfo["numnodes"], mask)),
        "water_profile": list(compress(nodeloopinfo["water_profile"], mask)),
        "poi_diversity": list(compress(nodeloopinfo["poi_diversity"], mask)),
    }
