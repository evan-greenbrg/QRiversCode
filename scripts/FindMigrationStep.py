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

river = 'Cowlitz'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/MigrationTstep'
# Migration Paths 
inname = '*migration_df.csv'
inpath = os.path.join(root, inname)
fps = glob.glob(inpath, recursive=True)

dfs = {}
for fp in fps:
    year = fp.split('/')[-1].split('_')[0]
    print(year)
    dfs[year] = pandas.read_csv(fp)

years = sorted(list(dfs.keys()))
ratio_centerline = []
ratio_poly = []
for year in years:
    df = dfs[year]
    maxcenter = df['migration_centerline_m'].quantile(.95)
    maxpoly = df['migration_poly_m'].quantile(.95)
    medwidth = df['width_m'].quantile(.5)

    if not maxcenter:
        ratio_centerline.append(None)
    else:
        ratio_centerline.append(medwidth / maxcenter)
    if not maxpoly:
        ratio_poly.append(None)
    else:
        ratio_poly.append(medwidth / maxpoly)

    print(year)

    

plt.plot(years, ratio_centerline)
plt.plot(years, ratio_poly)
plt.show()
