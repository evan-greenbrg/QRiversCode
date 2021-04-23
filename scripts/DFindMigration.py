import os
import pickle
import glob
import re

import rasterio
import pandas
import numpy as np
from shapely import geometry
from scipy import spatial
import geopandas as gpd
from matplotlib import pyplot as plt
import networkx as nx

from PyRivers import Width
from PyRivers import Centerline
from PyRivers.intersect import intersection
from PyRivers.GraphSort import getGraph, GraphSort 
from PyRivers.Migration import coordToIndex, pickCutoffs, channelMigrationPoly, channelMigrationCenterline
from PyRivers.Curvature import getCurvature


pattern = '(.*)\/(\w*)\/.*\/(\d{4})\/(\w*)\/'

river = 'Chehalis'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/**'
inname = '*polygon'

inpath = os.path.join(root, inname)
fps = glob.glob(inpath, recursive=True)

# Load in meta data for each year
data = {
    'root': [],
    'river': [],
    'year': [],
    'idx': [],
    'poly': [],
    'fp': [],
}
for fp in fps:
    regex = re.search(pattern, fp)
    data['root'].append(regex.group(1))
    data['river'].append(regex.group(2))
    data['year'].append(regex.group(3))
    data['idx'].append(regex.group(4))
    data['fp'].append(fp)

    with open(fp, "rb") as poly_file:
        data['poly'].append(pickle.load(poly_file))

meta_df = pandas.DataFrame(data)
meta_df = meta_df.sort_values(by='year').reset_index(drop=True)

# Get river polygons for each year
polys = {}
for idx, row in meta_df.iterrows():
    polys[row['year']] = row['poly']

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

# Image Paths 
inname = '*centerline.tif'
inpath = os.path.join(root, inname)
image_fps = glob.glob(inpath, recursive=True)

image_data = {
    'root': [],
    'river': [],
    'year': [],
    'idx': [],
    'fp': [],
}
for fp in image_fps:
    regex = re.search(pattern, fp)
    image_data['root'].append(regex.group(1))
    image_data['river'].append(regex.group(2))
    image_data['year'].append(regex.group(3))
    image_data['idx'].append(regex.group(4))
    image_data['fp'].append(fp)
image_meta = pandas.DataFrame(
    image_data
).sort_values(
    by='year'
).reset_index(drop=True)

images = {}
for idx, row in image_meta.iterrows():
    images[row['year']] = row['fp']
image_meta = None

# Graph Paths 
inname = '*graph'
inpath = os.path.join(root, inname)
graph_fps = glob.glob(inpath, recursive=True)
graphs = {}
for fp in graph_fps:
    year = fp.split('/')[-1].split('_')[1]
    graphs[year] = nx.read_gpickle(fp)

years = list(polys.keys())
years_p1 = np.roll(years, -1)
year_pairs = [
     ('1984', '1988'),
     ('1984', '1992'),
     ('1984', '1996'),
     ('1984', '2000'),
     ('1984', '2004'),
     ('1984', '2008'),
     ('1984', '2013'),
     ('1984', '2016'),
     ('1984', '2020'),
]

for year1, year2 in year_pairs:
    print(f'Running {year1}')
#    if abs(int(year2) - int(year1)) > 3:
#        continue

    # Get pixel conversion
    print('Getting conversion')
    ds = rasterio.open(images[year2])
    conversion = (111.32/.001) * ds.transform[0]
    ds = None

    print('Finding Statistics')
    # Load relevant data
    one = polys[year1]
    two = polys[year2]
    centerline1 = centerlines[year1]
    centerline2 = centerlines[year2]
    graph1 = graphs[year1]
    graph2 = graphs[year2]
    image1 = images[year1]
    image2 = images[year2]

    # Pick cutoff Points
    print('Pick cutoff points')
    p1 = gpd.GeoSeries(one)
    p2 = gpd.GeoSeries(two)
    cutoffs, cutoffsI = pickCutoffs(p1, p2, centerline1, centerline2, graph2)

    # Find curvature
    centerline2['curvature'] = getCurvature(graph2, centerline2)
    print(centerline2['curvature'])

    # Get two time periods for migration 
    centerlineSort1 = GraphSort(
        graph1,
        centerline1,
    )

    centerlineSort2 = GraphSort(
        graph2,
        centerline2,
    )

    # Need to match the sorted indices 
    centerline2['migration_centerline'] = None
    centerlineSort2['migration_centerline'] = channelMigrationCenterline(
        centerlineSort1,
        centerlineSort2,
        centerline2['width'].mean()
    )

    for idx, row in centerlineSort2.iterrows():
        centerline2.at[
            row['idx'],
            'migration_centerline'
        ] = row['migration_centerline']

    print('Get channel migration from differencing')
    centerline2['migration_poly'] = channelMigrationPoly(
        one, 
        two, 
        centerline1, 
        centerline2, 
        centerline2['width'].mean()
    )

    # Remove cutoff points
    print('Clean dataframe')
    centerline2['span'] = int(year2) - int(year1)
    centerline2['cutoff'] = 0

    if len(cutoffs):
        centerline2.at[cutoffs['index'], 'cutoff'] = cutoffs['cutoff']
        centerline2.at[cutoffs['index'], 'migration_centerline'] = None
        centerline2.at[cutoffs['index'], 'migration_poly'] = None

    # Convert to m
    print('Convert migration to meters')
    centerline2['migration_centerline_m'] = (
        centerline2['migration_centerline'] * conversion
    )
    centerline2['migration_poly_m'] = (
        centerline2['migration_poly'] * conversion
    )

    # Save dataframe
    print('Save dataframe')
    out_path = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/migration'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_name = f'{year2}_migration_df.csv'
    out = os.path.join(out_path, out_name)
    centerline2.to_csv(out)



# centerline2['MrC'] = centerline2['migration_centerline_m'] / centerline2['span']
# centerline2['MrP'] = centerline2['migration_poly_m'] / centerline2['span']
