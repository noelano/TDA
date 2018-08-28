"""
@author: Noel
@date: {13/08/2018}
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, cm
import kmapper as km
import graphviz as gz
import itertools
import colorsys
import sklearn
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import igraph as ig

np.random.seed(1234)

# Lod dataset
df = pd.read_csv("DetailedPlayerStats.csv", encoding='ANSI')

# Remove redundant features
to_drop = ['Id', 'FirstName', 'SecondName', 'DreamTeam', 'SelectedBy',
           'Position', 'Team', 'FixtureWeek1', 'Opponent1', 'HomeAway1',
           'FixtureWeek2', 'Opponent2', 'HomeAway2', 'FixtureWeek3', 'Opponent3',
           'HomeAway3', 'GameWeek', 'Cost', 'NetTransfers']
features = [col for col in df.columns if col not in to_drop]
df = df[features]

# Get average stats per game per player
g = df.groupby(u'WebName')
df2 = g.aggregate(np.mean)

# Filter out players who didn't play
df2 = df2[df2['MinutesPlayed'] > 0]

# Drop fantasy related clounms - we only want to cluster players based on their in-game performance
X = df2[[col for col in df2.columns if col not in ['Points', 'Bonus']]]
names = X.index.values
X.index = [i for i in range(X.shape[0])]

# Initialise mapper and create lens using TSNE
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X.values, projection=sklearn.manifold.TSNE(), scaler=None)

# Create the graph of the nerve of the corresponding pullback
graph = mapper.map(lens, X.values,
                   # clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=1),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1234),
                   nr_cubes=30, overlap_perc=0.9)

# Get the players per cluster
node_dict = {}
node_list = []
for node in graph['nodes']:
    node_list.append(node)
    players = [names[i] for i in graph['nodes'][node]]
    node_dict[node] = players

"""
# Use graphviz to create a custom plot
network = gz.Graph()
network.attr(layout='neato')
network.attr('edge', len='0.7')
network.attr('node', shape='circle', fixedsize='true', width='0.3',
             label=None, fontsize='9', style='filled', fontcolor='white')

for node in graph['links']:
    for nbr in graph['links'][node]:
        # We'll remove the 'cube' and 'cluster' strings from the label to avoid clutter
        node = str.replace(str.replace(node, 'cube', ''), 'cluster', '')
        nbr = str.replace(str.replace(nbr, 'cube', ''), 'cluster', '')
        network.edge(node, nbr)

# Colour the nodes based on the average x value of points in the cluster
norm = colors.Normalize(vmin=df2['Points'].min(), vmax=df2['Points'].max())
clrmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

for node in graph['nodes']:
    avg_pts = np.average([df2.iloc[i]['Points'] for i in graph['nodes'][node]])
    color = clrmap.to_rgba(avg_pts)
    color = colorsys.rgb_to_hsv(color[0], color[1], color[2])
    node = str.replace(str.replace(node, 'cube', ''), 'cluster', '')
    network.node(node, fillcolor="%f, %f, %f" % (color[0], color[1], color[2]))

# Display the graph
# network.render('gz_plot.pdf', view=True)

mapper.visualize(graph, path_html="premier_league_players_2.html",
                 title="Player Data", inverse_X=X.values, inverse_X_names=names, custom_tooltips=names)
"""

# Create igraph viz

# Add the edges:
edge_list = []
for node in graph['links']:
    for nbr in graph['links'][node]:
        # Need to base everything on indices for igraph
        edge_list.append((node_list.index(node), node_list.index(nbr)))

n_nodes = len(node_list)
G = ig.Graph(n_nodes)

G.add_edges(edge_list)

avg_points = []
for node in G.vs.indices:
    avg_points.append(np.average([df2.iloc[i]['Points'] for i in graph['nodes'][node_list[node]]]))

links = G.get_edgelist()
layt = G.layout('kk')

N = len(layt)
Xnodes = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
Ynodes = [layt[k][1] for k in range(N)]  # y-coordnates of nodes

Xedges = []
Yedges = []
for e in links:
    Xedges.extend([layt[e[0]][0], layt[e[1]][0], None])
    Yedges.extend([layt[e[0]][1], layt[e[1]][1], None])

edge_linecolor = 'rgb(200,200,200)'
edge_linewidth = 1.5
colorscale = 'Viridis'
reversescale = False
showscale = True

edges_trace = dict(type='scatter',
                   x=Xedges,
                   y=Yedges,
                   mode='lines',
                   line=dict(color=edge_linecolor,
                             width=edge_linewidth),
                   hoverinfo='none')

nodes_trace = dict(type='scatter',
                   x=Xnodes,
                   y=Ynodes,
                   mode='markers',
                   marker=dict(symbol='dot',
                               colorscale=colorscale,
                               showscale=showscale,
                               reversescale=reversescale,
                               color=avg_points,
                               colorbar=dict(thickness=20,
                                             ticklen=4)),
                   text=[],
                   hoverinfo='text')

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

# Add tooltips
nodes = G.vs.indices

for node in nodes:
    node_name = node_list[node]
    players = node_dict[node_name]
    node_info = node_name + '<br>-------<br>' + '<br>'.join(players)
    nodes_trace['text'] += tuple([node_info])

plot(dict(data=[edges_trace, nodes_trace], layout=layout))

