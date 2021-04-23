import glob
import os
import re

import rasterio
from matplotlib import pyplot as plt
from PyRivers import Centerline 

pattern = '(.*)\/(\w*)\/.*\/(\d{4})\/(\w*)\/'

river = 'Chehalis'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/**'
inname = '*mask.tif'

inpath = os.path.join(root, inname)
fps = glob.glob(inpath, recursive=True)

es = 'EW'

for i, fp in enumerate(fps):
    print(fp)
    # Find components of the path
    regex = re.search(pattern, fp)
    root = regex.group(1)
    river = regex.group(2)
    year = regex.group(3)
    idx = regex.group(4)

    # Open the image
    ds = rasterio.open(fp)
    image = ds.read()
    image = image[0, :, :]

    image = Centerline.getLargest(image)
    image = Centerline.fillHoles(image)
    raw_centerline = Centerline.getCenterline(image)

    thresh = 100
    close_points = False 
    while not close_points:
        print('Trying threshold: ', thresh)
        centerline, river_endpoints = Centerline.cleanCenterline(
            raw_centerline, 
            es, 
            thresh 
        )
        centerline = Centerline.getLargest(centerline)

        close_points = Centerline.getCenterlineExtent(
            centerline, 
            river_endpoints,
            maxdistance=50
        )
        print(close_points)
        thresh -= 10

    # Use the found threshold
    thresh = thresh + 10
    print('Using threshold: ', thresh)
    centerline, river_endpoints = Centerline.cleanCenterline(
        raw_centerline, 
        es, 
        thresh
    )
    centerline = Centerline.getLargest(centerline)

    # Save the centerline image
    oroot = os.path.join(
        root,
        river,
        'centerline',
        year,
        idx
    )

    oname = f'{river}_{year}_1_centerline.tif'
    opath = os.path.join(oroot, oname)

    if not os.path.exists(oroot):
        os.makedirs(oroot)

    meta = ds.meta.copy()
    meta.update({'dtype': rasterio.int8, 'count': 1})
    with rasterio.open(opath, "w", **meta) as dest:
        dest.write(centerline.astype(rasterio.int8), 1)

