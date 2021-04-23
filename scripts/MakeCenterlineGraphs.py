import os
import pickle
import glob
import re

import rasterio
import networkx as nx
import pandas
import numpy as np
from shapely import geometry
from scipy import spatial
import geopandas as gpd
from matplotlib import pyplot as plt
from PyRivers.GraphSort import getGraph, GraphSort 

pattern = '(.*)\/(\w*)\/.*\/(\d{4})\/(\w*)\/'

river = 'Chehalis'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/**'
# Get centerlines
inname = '*width.csv'
inpath = os.path.join(root, inname)
centerline_fps = glob.glob(inpath, recursive=True)

centerline_data = {
    'root': [],
    'river': [],
    'year': [],
    'idx': [],
    'fp': [],
}
for fp in centerline_fps:
    regex = re.search(pattern, fp)
    centerline_data['root'].append(regex.group(1))
    centerline_data['river'].append(regex.group(2))
    centerline_data['year'].append(regex.group(3))
    centerline_data['idx'].append(regex.group(4))
    centerline_data['fp'].append(fp)

centerline_meta = pandas.DataFrame(
    centerline_data
).sort_values(
    by='year'
).reset_index(drop=True)

centerlines = {}
for idx, row in centerline_meta.iterrows():
    centerlines[row['year']] = pandas.read_csv(row['fp'])

centerline_meta = None
years = list(centerlines.keys())

# Make graphs
graphs = {}
for year in years:
    print(year)
    centerline = centerlines[year]
    graphs[year] = getGraph(
        centerline, 
        start=0,
        end=len(centerline)-1,
        xcol='rowi', 
        ycol='coli'
    )

root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/graphs'
for year, G in graphs.items():
    print(year)
    name = f'{river}_{year}_graph'
    opath = os.path.join(root, name)
    nx.write_gpickle(G, opath)
