def NormalizeData(data):
    return list((data - np.min(data)) / (np.max(data) - np.min(data)))

def getLayout(G, nodes_id, nodes_coords):
    named_vertex_list = G.vs()["name"]
    layout = []
    for n in named_vertex_list:
        pos = nodes_id.index(n)
        layout.append(nodes_coords[pos])
    return layout

def plotCheck(G, nodes_id, nodes_coords, vertex_size = 7, edge_color=None):
    fig, ax = plt.subplots()
    layout = getLayout(G, nodes_id, nodes_coords)
    if edge_color is None:
        ig.plot(G, target=ax, vertex_size=vertex_size, layout=layout, vertex_color="green");
    else:
        ig.plot(G, target=ax, vertex_size=vertex_size, layout=layout, edge_color=edge_color, vertex_color="green");
    return fig
    
def getLoopLength(c):
    l = 0
    cl = len(c)
    for i in range(cl):
        l += Gnx[c[i%cl]][c[(i+1)%cl]]["weight"]
    return l

def getLoopMaxSlope(c):
    ms = 0
    cl = len(c)
    for i in range(cl):
        ms = max([ms, Gnx[c[i%cl]][c[(i+1)%cl]]["max_slope"]])
    return ms

def getLoopWaterProfile(c):
    wp = tuple()
    cl = len(c)
    l = 0
    for i in range(cl):
        l_this = Gnx[c[i%cl]][c[(i+1)%cl]]["weight"]
        l += l_this
        if Gnx[c[i%cl]][c[(i+1)%cl]]["has_water"]: # If link has water, assume it is half the link length in
            wp += (l+l_this/2,)
            l = l_this/2
    return wp