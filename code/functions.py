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
                vertex_sizes.append(np.log2(len(loopinfo[k]["loops"]) + 1.00001))
            else:
                # if linear, make it the marker's area, so take the sqrt
                vertex_sizes.append(math.sqrt(len(loopinfo[k]["loops"])))
        except:
            vertex_sizes.append(0)

    numloops_max = max(vertex_sizes)
    vertex_sizes = [i / (numloops_max / max_node_size) for i in vertex_sizes]
    return vertex_sizes, numloops_max


def get_vertex_plotinfo(loopinfo, max_node_size=20):
    """
    Calculate a node size and color for each node in the loopinfo
    dict for plotting given the number of loops in the node. The
    largest value gets size max_node_size.
    """
    cmap = mpl.colormaps["viridis"].resampled(8)
    cmaparr = cmap(np.linspace(0, 1, 8))
    # cmaparr = np.vstack((cmaparr, np.repeat(cmaparr[-1:, :], 10, axis=0)))
    cmaparr = np.vstack(
        (cmaparr, np.repeat([[0.871, 0.176, 0.149, 1]], 10, axis=0))
    )  # white

    vertex_sizes = []
    vertex_colors = np.zeros((len(loopinfo.keys()), 4))
    for k in range(len(loopinfo.keys())):
        try:
            if PLOTLOGSCALE:
                val = np.clip(math.ceil(np.log2(len(loopinfo[k]["loops"]))), 0, 9)
                vertex_sizes.append(math.sqrt(val))
                vertex_colors[k, :] = cmaparr[val - 1, :]
            else:
                # if linear, make it the marker's area, so take the sqrt
                vertex_sizes.append(math.sqrt(len(loopinfo[k]["loops"])))
        except:
            vertex_sizes.append(0)

    vertex_sizes = [i / (18 / max_node_size) for i in vertex_sizes]
    return vertex_sizes, vertex_colors


def lighten_color(color, amount=0.5):
    # Source: https://stackoverflow.com/a/49601444
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(
        np.clip(c[0], 0, 1), np.clip(1 - amount * (1 - c[1]), 0, 1), np.clip(c[2], 0, 1)
    )


def get_layout(G, nodes_id, nodes_coords, mode="single"):
    if mode == "combined":
        return nodes_coords
    else:
        named_vertex_list = G.vs()["name"]
        layout = []
        for n in named_vertex_list:
            try:
                pos = nodes_id.index(n)
            except:
                print("There was an invalid node with name: " + str(n))
            layout.append(nodes_coords[pos])
        return layout


def plot_dk_gdf(
    nodes,
    edges,
    scale=1,
    vertex_size=7,
    vertex_color="#009F92",
    link_width=2,
    link_color="#009F92",
):
    fig = plt.figure(
        figsize=(scale * 640 / PLOTPARAM["dpi"], scale * 760 / PLOTPARAM["dpi"]),
        dpi=PLOTPARAM["dpi"],
    )
    ax = fig.add_axes(
        [-0.03, -0.03, 1.06, 1.06]
    )  # negative because plot() introduces a padding
    nodes.plot(
        ax=ax,
        markersize=vertex_size,
        alpha=1,
        color=vertex_color,
        edgecolor="#666666",
        linewidth=0.6,
    )
    edges.plot(ax=ax, zorder=0, linewidth=link_width, color=link_color)
    ax.set_axis_off()


def plot_check(
    G,
    nodes_id,
    nodes_coords,
    vertex_size=7,
    edge_width=2,
    edge_color=None,
    mode="single",
):
    fig, ax = plt.subplots()
    layout = get_layout(G, nodes_id, nodes_coords, mode)
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
    # To do: Possibly improve speed by using binary numbers
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


### TOPOLOGICAL EVALUATION
# Source: src/eval_func.py and src/plot_func.py
# in https://github.com/anastassiavybornova/bike-node-planner


def classify_edgelength(length_km, ideal_length_lower, ideal_length_upper, max_length):
    """
    length_km: length in km
    ideal_length_lower, ideal_length upper: lower and upper threshold for length's ideal range
    max_length: maximum tolerable length
    """
    assert (ideal_length_lower < ideal_length_upper) and (
        ideal_length_upper < max_length
    ), "Please provide valid length ranges"

    if length_km < ideal_length_lower:
        classification = "below_ideal"
    elif length_km < ideal_length_upper:
        classification = "ideal_range"
    elif length_km < max_length:
        classification = "above_ideal"
    else:
        classification = "too_long"

    return classification


def classify_looplength(length_km, loop_length_min, loop_length_max):
    """
    length_km: length in km
    ideal_length_lower, ideal_length upper: lower and upper threshold for length's ideal range
    max_length: maximum tolerable length
    """
    assert loop_length_min < loop_length_max, "Please provide valid length ranges"

    if length_km < loop_length_min:
        classification = "below_ideal"
    elif length_km < loop_length_max:
        classification = "ideal_range"
    else:
        classification = "too_long"

    return classification


def classify_maxslope(maxslope, maxslope_medium, maxslope_hard):
    """
    maxslope: maximum gradient in %
    """
    assert maxslope_medium < maxslope_hard, "Please provide valid maxslope ranges"

    if maxslope < maxslope_medium:
        classification = "easy"
    elif maxslope < maxslope_hard:
        classification = "medium"
    else:
        classification = "hard"

    return classification
