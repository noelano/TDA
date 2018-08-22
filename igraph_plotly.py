"""
@author: Noel
@date: {22/08/2018}
"""

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import igraph as ig


def get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates assigned by igraph.Layout
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]  # x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]  # y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def plotly_graph(G, graph_layout='kk', colorscale='Viridis',
                 reversescale=False, showscale=True, factor_size=2,
                 keep_kmtooltips=True,
                 edge_linecolor='rgb(200,200,200)', edge_linewidth=1.5):
    # kmgraph: a dict returned by the method visualize, when path_html=None
    # graph_layout: an igraph layout; recommended 'kk' or 'fr'
    # factor_size: a factor for the node size
    # keep_tooltip: True  to keep the tooltips assigned by kmapper;
    # False, when kmapper tooltips contains projection statistic

    links = G.get_edgelist()
    layt = G.layout(graph_layout)

    color_vals = None
    # size = np.array([factor_size * node['size'] for node in kmgraph['nodes']],
    # dtype=np.int)
    Xn, Yn, Xe, Ye = get_plotly_data(links, layt)
    edges_trace = dict(type='scatter',
                       x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color=edge_linecolor,
                                 width=edge_linewidth),
                       hoverinfo='none')

    nodes_trace = dict(type='scatter',
                       x=Xn,
                       y=Yn,
                       mode='markers',
                       marker=dict(symbol='dot',
                                   colorscale=colorscale,
                                   showscale=showscale,
                                   reversescale=reversescale,
                                   color=[],
                                   colorbar=dict(thickness=20,
                                                 ticklen=4)),
                       text=[],
                       hoverinfo='text')

    return [edges_trace, nodes_trace]


axis = dict(showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')

layout = dict(title='Test',
              font=dict(size=12),
              showlegend=False,
              autosize=False,
              width=700,
              height=700,
              xaxis=dict(axis),
              yaxis=dict(axis),
              hovermode='closest',
              plot_bgcolor='rgba(20,20,20, 0.8)')


# define an igraph.Graph instance of n_nodes
G = ig.Graph.GRG(200, 0.125)
edges = G.get_edgelist()
nodes = G.vs.indices

edges_trace, node_trace = plotly_graph(G)

for node in nodes:
    node_trace['marker']['color'] += [G.degree(node)]
    node_info = '# of connections: '+str(G.degree(node))
    node_trace['text'] += tuple([node_info])

plot(dict(data=[edges_trace, node_trace], layout=layout))
