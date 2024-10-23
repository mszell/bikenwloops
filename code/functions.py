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


def rgb2hex(rgb_string):
    return "#%02x%02x%02x" % tuple([int(n) for n in rgb_string.split(",")])[0:3]


def plot_edge_lengths(homepath, edge_classification_colors):
    topo_folder = homepath + "/data/output/network/topology/"

    config = yaml.load(
        open(homepath + "/config/config-topological-analysis.yml"),
        Loader=yaml.FullLoader,
    )
    [ideal_length_lower, ideal_length_upper] = config["ideal_length_range"]
    max_length = config["max_length"]

    edge_classification_labels = {
        "below_ideal": f"(<{ideal_length_lower}km)",
        "ideal_range": f"({ideal_length_lower}-{ideal_length_upper}km)",
        "above_ideal": f"({ideal_length_upper}-{max_length}km)",
        "too_long": f"(>{max_length}km)",
    }

    gdf = gpd.read_file(topo_folder + "edges_length_classification.gpkg")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for classification in gdf.length_class.unique():
        gdf[gdf.length_class == classification].plot(
            ax=ax,
            color=edge_classification_colors[classification],
            label=classification.replace("_", " ")
            + " "
            + edge_classification_labels[classification],
        )
    ax.legend()
    ax.set_title("Edge length evaluation")
    ax.set_axis_off()
    fig.savefig(
        homepath + f"/results/plots/edgelengths.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return None


def plot_loop_lengths(homepath, loop_classification_colors):
    topo_folder = homepath + "/data/output/network/topology/"

    config = yaml.load(
        open(homepath + "/config/config-topological-analysis.yml"),
        Loader=yaml.FullLoader,
    )

    [loop_length_min, loop_length_max] = config["loop_length_range"]

    loop_classification_labels = {
        "below_ideal": f"(<{loop_length_min}km)",
        "ideal_range": f"({loop_length_min}-{loop_length_max}km)",
        "too_long": f"(>{loop_length_max}km)",
    }

    gdf = gpd.read_file(topo_folder + "loops_length_classification.gpkg")

    gdf["color_plot"] = gdf.length_class.apply(
        lambda x: rgb2hex(loop_classification_colors[x])
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(ax=ax, color=gdf.color_plot, alpha=0.5)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linestyle="dashed")
    ax.set_title("Loop length evaluation")
    ax.set_axis_off()
    # add custom legend
    custom_lines = [
        Line2D([0], [0], color=rgb2hex(k), lw=4, alpha=0.5)
        for k in loop_classification_colors.values()
    ]
    ax.legend(custom_lines, loop_classification_labels.values())
    fig.savefig(
        homepath + f"/results/plots/looplengths.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return None
