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
for node in graph['nodes']:
    players = [names[i] for i in graph['nodes'][node]]
    node_dict[node] = players

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
