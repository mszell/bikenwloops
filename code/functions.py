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


def get_vertex_sizes(loopinfo, max_node_size=20, plotfunc="log2"):
    """
    Calculate a node size for each node in the loopinfo dict
    for plotting given the number of loops in the node. The
    largest value gets size max_node_size.
    Returns a list of node sizes (floats).
    """
    vertex_sizes = []
    for k in range(len(loopinfo.keys())):
        try:
            if plotfunc == "log2":
                vertex_sizes.append(np.log2(len(loopinfo[k]["loops"]) + 1.00001))
            elif plotfunc == "sqrt":
                # if linear, make it the marker's area, so take the sqrt
                vertex_sizes.append(math.sqrt(len(loopinfo[k]["loops"])))
            else:  # else just return the raw number
                vertex_sizes.append(len(loopinfo[k]["loops"]))
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


def get_vertex_plotinfo(
    loopinfo, max_node_size=150, bit_threshold=8, maxbits=18, plotfunc="log2"
):
    """
    Calculate a node size and color for each node in the loopinfo
    dict for plotting given the number of loops in the node. The
    node sizes are fixed, max_node_size just scales them.
    bit_threshold is the threshold from which nodes are colored red.
    """
    cmaparr = get_cmap(bit_threshold=bit_threshold)

    vertex_sizes = []  # list
    vertex_colors = np.zeros((len(loopinfo.keys()), 4))  # numpy array
    for i, k in enumerate(loopinfo.keys()):
        try:
            # make the value correspond to the disk's area, so take the sqrt
            if plotfunc == "log2":
                val = np.clip(
                    math.ceil(np.log2(len(loopinfo[k]["loops"]))), 0, bit_threshold + 1
                )
                vertex_sizes.append(math.sqrt(val) + 2)
                vertex_colors[i, :] = cmaparr[val, :]
            else:
                vertex_sizes.append(math.sqrt(len(loopinfo[k]["loops"])))
        except:
            val = 0
            vertex_sizes.append(math.sqrt(val) + 2)
            vertex_colors[i, :] = PLOTPARAM["color"]["noloop"]

    vertex_sizes = [(i - 1.25) / (maxbits / max_node_size) for i in vertex_sizes]
    return vertex_sizes, vertex_colors


def get_vertex_loopnums(loopinfo, func=""):
    numvertices = len(loopinfo.keys())
    vals = np.zeros(numvertices)
    for i in range(numvertices):
        try:
            if func == "log2":
                vals[i] = math.log2(len(loopinfo[i]["loops"]) + 1)
            else:
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


def plot_dk_inset(fig, loopinfo, bit_threshold=8, ymaxconst=7800):
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
        (max(N) + n_upper_outliers) * 1.1 if not ymaxconst else ymaxconst * 0.985,
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
        xmax - 0.1,
        (max(N) + n_upper_outliers) * 1.1 if not ymaxconst else ymaxconst * 0.985,
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
    # axes.set_ylim([0, 1.08 * axes.get_ylim()[1]])
    if not ymaxconst:
        axes.set_ylim([0, 1.12 * (max(N) + n_upper_outliers)])
    else:
        axes.set_ylim([0, ymaxconst])


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


def mask_node(
    nodeloopinfo,
    mask,
    keys=["loops", "lengths", "numnodes", "water_profile", "poi_diversity"],
):
    return {k: list(compress(nodeloopinfo[k], mask)) for k in keys}


def synthnx_to_momepy(Gnx):
    Gnx.graph["approach"] = "primal"
    Gnx_nodes, Gnx_links = momepy.nx_to_gdf(net=Gnx, points=True, lines=True)
    for index, row in Gnx_nodes.iterrows():
        Gnx_nodes.at[index, "geometry"] = Point(row["pos"])
    Gnx_links["geometry"] = ""
    for index, row in Gnx_links.iterrows():
        Gnx_links.at[index, "geometry"] = LineString(
            [
                Gnx_nodes.at[row["node_start"], "geometry"],
                Gnx_nodes.at[row["node_end"], "geometry"],
            ]
        )
    Gnx_links.set_geometry("geometry", inplace=True)

    return Gnx_nodes, Gnx_links


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


def restrict_scenario(allloops_all, allloops_prev, level=0):
    """
    Restrict the loops in the dict allloops_prev by:
    level 0: scenario length
    level 1: scenario gradients
    level 2: water limits
    level 3: POI diversity
    """

    allloops_next = {}

    if level == 0:
        for sourcenode in tqdm(allloops_prev, desc="Restrict to scenario lengths"):
            try:
                lengths_this = allloops_all[sourcenode]["lengths"] * MPERUNIT
                mask_this = (lengths_this >= SCENARIO[SCENARIOID]["looplength_min"]) & (
                    lengths_this <= SCENARIO[SCENARIOID]["looplength_max"]
                )
                allloops_next[sourcenode] = mask_node(
                    allloops_prev[sourcenode], mask_this
                )
            except:  # Account for 0 loop nodes
                allloops_next[sourcenode] = {}

    elif level == 1:
        for sourcenode in tqdm(allloops_prev, desc="Restrict to scenario gradients"):
            try:
                lengths_this = allloops_all[sourcenode]["lengths"] * MPERUNIT
                maxslopes_this = (
                    allloops_all[sourcenode]["max_slopes"] / 100.0
                )  # max_slopes were multiplied by 100 for storage as uint16
                mask_this = lengths_this >= SCENARIO[SCENARIOID]["looplength_min"]
                mask_this &= lengths_this <= SCENARIO[SCENARIOID]["looplength_max"]
                mask_this &= maxslopes_this <= SCENARIO[SCENARIOID]["maxslope_limit"]
                allloops_next[sourcenode] = mask_node(
                    allloops_all[sourcenode], mask_this
                )
            except:  # Account for 0 loop nodes
                allloops_next[sourcenode] = {}

    elif level == 2:
        for sourcenode in tqdm(allloops_prev, desc="Restrict to water limits"):
            try:
                numloops = len(allloops_prev[sourcenode]["loops"])
                mask_this = [True] * numloops
                for i in range(numloops):
                    wp = allloops_prev[sourcenode]["water_profile"][i]
                    water_enough = True
                    if wp:  # There is water on the way somewhere. Check distances
                        for w in wp:
                            if w > WATERLENGTH_MAX:
                                water_enough = False
                                break
                        if water_enough and (
                            allloops_prev[sourcenode]["lengths"][i] - wp[-1]
                            > WATERLENGTH_MAX
                        ):
                            water_enough = False
                    else:  # No water on the way, so the loop is only valid if short enough
                        if allloops_prev[sourcenode]["lengths"][i] > WATERLENGTH_MAX:
                            water_enough = False
                    mask_this[i] = water_enough
                allloops_next[sourcenode] = mask_node(
                    allloops_prev[sourcenode], mask_this
                )
            except:  # Account for 0 loop nodes
                allloops_next[sourcenode] = {}

    elif level == 3:
        for sourcenode in tqdm(allloops_prev, desc="Restrict with POI diversity"):
            try:
                numloops = len(allloops_prev[sourcenode]["loops"])
                mask_this = [False] * numloops
                for i in range(numloops):
                    poidiv = allloops_prev[sourcenode]["poi_diversity"][i]
                    if poidiv >= 3:
                        mask_this[i] = True
                allloops_next[sourcenode] = mask_node(
                    allloops_prev[sourcenode], mask_this
                )
            except:  # Account for 0 loop nodes
                allloops_next[sourcenode] = {}

    return allloops_next


# https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7")

    return ax


def equation(a, b):
    return np.polyval(a, b)


def confband(x, y, ax):
    p, cov = np.polyfit(
        x, y, 1, cov=True
    )  # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(
        p, x
    )  # model using the fit parameters; NOTE: parameters here are coefficients

    # Statistics
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = scipy.stats.t.ppf(0.975, n - m)  # t-statistic; used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model  # residuals; diff. actual data from predicted values
    chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = equation(p, x2)

    # Confidence Interval (select one)
    plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
tinv = lambda alpha, df: abs(scipy.stats.t.ppf(alpha / 2, df))
