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


def get_cmap(maxbits=18, bit_threshold=8):
    cmap = mpl.colormaps["viridis"].resampled(bit_threshold)
    cmaparr = cmap(np.linspace(0, 1, bit_threshold))
    cmaparr = np.vstack(
        (
            cmaparr,
            np.repeat([[0.871, 0.176, 0.149, 1]], maxbits - bit_threshold, axis=0),
        )
    )  # red: #de2d26
    return cmaparr


def get_vertex_plotinfo(loopinfo, max_node_size=150, bit_threshold=8, maxbits=18):
    """
    Calculate a node size and color for each node in the loopinfo
    dict for plotting given the number of loops in the node. The
    node sizes are fixed, max_node_size just scales them.
    bit_threshold is the threshold from which nodes are colored red.
    """
    cmaparr = get_cmap(bit_threshold=bit_threshold)

    vertex_sizes = []  # list
    vertex_colors = np.zeros((len(loopinfo.keys()), 4))  # numpy array
    for k in range(len(loopinfo.keys())):
        try:
            # make the value correspond to the disk's area, so take the sqrt
            if PLOTLOGSCALE:
                val = np.clip(
                    math.ceil(np.log2(len(loopinfo[k]["loops"]))), 0, bit_threshold + 1
                )
                vertex_sizes.append(math.sqrt(val) + 2)
                vertex_colors[k, :] = cmaparr[val, :]
            else:
                vertex_sizes.append(math.sqrt(len(loopinfo[k]["loops"])))
        except:
            val = 0
            vertex_sizes.append(math.sqrt(val) + 2)
            vertex_colors[k, :] = PLOTPARAM["color"]["noloop"]

    vertex_sizes = [(i - 1.25) / (maxbits / max_node_size) for i in vertex_sizes]
    return vertex_sizes, vertex_colors


def get_vertex_loopnums(loopinfo):
    numvertices = len(loopinfo.keys())
    vals = np.zeros(numvertices)
    for i in range(numvertices):
        try:
            vals[i] = len(loopinfo[i]["loops"])
        except:
            pass
    return vals


def get_link_plotinfo(
    edges, var_bad="max_slope", var_good=None, threshold_bad=4, threshold_good=0
):
    """
    Calculate a link width and color for each link in the loopinfo
    dict for plotting given some criterion.
    """

    link_widths = np.zeros(len(edges))
    link_colors = np.zeros((len(edges), 4))
    if var_good is None:
        for i, vbad in enumerate(edges[var_bad]):
            if vbad < threshold_bad:
                link_widths[i] = 0.3
                link_colors[i, :] = [0.267, 0.267, 0.267, 1]
            else:
                link_widths[i] = PLOTPARAM["maxslope_classification_linewidths"][
                    "medium"
                ]
                link_colors[i, :] = [0.871, 0.176, 0.149, 1]
    else:
        for i, (vbad, vgood) in enumerate(zip(edges[var_bad], edges[var_good])):
            if vbad < threshold_bad and vgood >= threshold_good:  # all fulfilled
                link_widths[i] = PLOTPARAM["maxslope_classification_linewidths"][
                    "medium"
                ]
                link_colors[i, :] = [0.031, 0.271, 0.58, 1]  # blue
            elif vbad >= threshold_bad:  # bad
                link_widths[i] = PLOTPARAM["maxslope_classification_linewidths"][
                    "medium"
                ]
                link_colors[i, :] = [0.871, 0.176, 0.149, 1]
            else:  # not bad but also not good
                link_widths[i] = 0.3
                link_colors[i, :] = [0.267, 0.267, 0.267, 1]

    return link_widths, link_colors


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
    vertex_color=PLOTPARAM["color"]["dknt_green"],
    link_width=2,
    link_color=PLOTPARAM["color"]["dknt_green"],
):
    fig = plt.figure(
        figsize=(scale * 640 / PLOTPARAM["dpi"], scale * 760 / PLOTPARAM["dpi"]),
        dpi=PLOTPARAM["dpi"],
    )
    ax = fig.add_axes(
        [-0.03, -0.03, 1.06, 1.06]
    )  # negative because plot() introduces a padding
    if nodes is not None:
        nodes.plot(
            ax=ax,
            zorder=1,
            markersize=vertex_size,
            alpha=1,
            color=vertex_color,
            edgecolor=PLOTPARAM["color"]["neutral"],
            linewidth=0.6,
        )
    edges.plot(ax=ax, zorder=0, linewidth=link_width, color=link_color)

    bornholm_circle = shapely.Point(657_200, 6_245_800).buffer(23000).exterior
    bornholm_circle_gdf = gpd.GeoDataFrame(index=[0], geometry=[bornholm_circle])
    bornholm_circle_gdf.plot(
        ax=ax, zorder=2, linewidth=0.5, color=PLOTPARAM["color"]["neutral"]
    )

    ax.set_axis_off()
    return fig, ax


def plot_dk_scenariotext(ax, filterdepth=0):
    ax.text(
        0.06,
        0.925,
        SCENARIO[SCENARIOID]["name"],
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=11,
    )
    if filterdepth >= 1:
        ax.text(
            0.07,
            0.895,
            "Loop length ≥"
            + str(SCENARIO[SCENARIOID]["looplength_min"] // 1000)
            + "km and ≤"
            + str(SCENARIO[SCENARIOID]["looplength_max"] // 1000)
            + "km",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=9,
        )
    if filterdepth >= 2:
        ax.text(
            0.07,
            0.87,
            "Maximum gradient ≤" + str(SCENARIO[SCENARIOID]["maxslope_limit"]) + "%",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=9,
        )
    if filterdepth >= 3:
        ax.text(
            0.07,
            0.845,
            "Water every " + str(WATERLENGTH_MAX // 1000) + "km",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=9,
        )
    if filterdepth >= 4:
        ax.text(
            0.07,
            0.82,
            "POI diversity 3",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=9,
        )


def plot_dk_inset(fig, loopinfo, bit_threshold=8):
    xmax = bit_threshold + 2
    axes = fig.add_axes([0.64, 0.69, 0.31, 0.25])

    cmaparr = get_cmap(bit_threshold=bit_threshold)
    cmaparr = np.vstack((PLOTPARAM["color"]["noloop"], cmaparr))

    loopnums = get_vertex_loopnums(loopinfo)

    N, bins, patches = axes.hist(
        loopnums,
        bins=list(np.linspace(0, xmax, xmax + 1)),
        density=False,
        linewidth=0.5,
    )

    # Source: https://stackoverflow.com/a/49290555
    patches[0].set_edgecolor(PLOTPARAM["color"]["neutral"])
    for i in range(xmax):
        patches[i].set_facecolor(cmaparr[i, :])

    # Source: https://stackoverflow.com/a/51050772
    n_upper_outliers = (loopnums > bit_threshold).sum()
    patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)

    axes.text(
        0.1,
        (max(N) + n_upper_outliers) * 1.1,
        str(
            round(
                len([i for i, x in enumerate(loopnums) if (x == 0)])
                / len(loopnums)
                * 100
            )
        )
        + "%",
        horizontalalignment="left",
        verticalalignment="top",
        color=PLOTPARAM["color"]["neutral"],
    )

    axes.text(
        bit_threshold + 2,
        (max(N) + n_upper_outliers) * 1.1,
        str(
            round(
                len([i for i, x in enumerate(loopnums) if (x >= bit_threshold)])
                / len(loopnums)
                * 100
            )
        )
        + "%",
        horizontalalignment="right",
        verticalalignment="top",
        color=cmaparr[-1, :],
    )

    axes.set_xlabel("Bits $2^b$")
    axes.set_ylabel("Frequency")
    axes.set_title("Loops per node")
    axes.set_xticks([i + 0.5 for i in list(range(xmax))])
    axes.set_xticklabels(["No"] + [(str(i)) for i in list(range(xmax - 2))] + ["8+"])
    axes.set_xlim([0, xmax])
    axes.set_ylim([0, 1.08 * axes.get_ylim()[1]])
    axes.set_ylim([0, 1.12 * (max(N) + n_upper_outliers)])


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


def classify_length(length_km, ideal_length_lower, ideal_length_upper, max_length):
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
