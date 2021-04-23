import os
import pickle
import glob
import re

import rasterio
import pandas
import numpy
from shapely import geometry
from scipy.interpolate import interp1d
from PyRivers import Width


pattern = '(.*)\/(\w*)\/.*\/(\d{4})\/(\w*)\/'

river = 'Chehalis'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/**'
inname = '*centerline.tif'

inpath = os.path.join(root, inname)
fps = glob.glob(inpath, recursive=True)
print(fps)

#fps = [fps[0], fps[1]]
for i, fp in enumerate(fps):
    # Find components of the path
    regex = re.search(pattern, fp)
    root = regex.group(1)
    river = regex.group(2)
    year = regex.group(3)
    idx = regex.group(4)

    # Load centerline image
    ds = rasterio.open(fp)
    centerline = ds.read(1)

    # find the path of the channel mask
    maskpath = os.path.join(
        root,
        river,
        'mask',
        year,
        idx,
        f'{river}_{year}_1_mask.tif'
    )

    # Read mask image
    ids = rasterio.open(maskpath)
    image = ids.read(1)

    width_df, river_polygon = Width.getWidths(image, centerline, step=5)
    width_df = width_df.dropna(how='any')
    width_df = Width.getCoordinates(ds.transform, width_df)

    # Convert width form pixel to meters
    conversion = (111.32/.001) * ds.transform[0]
    width_df['width_m'] = width_df['width'] * conversion

    centerline_i = numpy.array(width_df[['rowi', 'coli']])
#     centerline_i = Width.sortCenterline(centerline_i)

    data = {
        'row': [],
        'col': [],
        'lat': [],
        'lon': [],
        'width': []
    }
    for idx, row in enumerate(centerline_i):
        match = width_df[
            (width_df['rowi'] == row[0])
            & (width_df['coli'] == row[1])
        ]
        data['row'].append(match['rowi'].iloc[0])
        data['col'].append(match['coli'].iloc[0])
        data['lat'].append(match['lat'].iloc[0])
        data['lon'].append(match['lon'].iloc[0])
        data['width'].append(match['width'])
    centerline_df = pandas.DataFrame(data)

    oroot = os.path.join(
        root,
        river,
        'width',
        year,
        str(idx)
    )
    oname = f'{river}_{year}_1_width.csv'
#    oname= f'{river}_centerline.csv'
    opath = os.path.join(oroot, oname)

    if not os.path.exists(oroot):
        os.makedirs(oroot)

    # Save the channel width CSV
    width_df.to_csv(opath)
#    centerline_df.to_csv(opath)

    oroot = os.path.join(
        root,
        river,
        'poly',
        year,
        str(idx)
    )
    oname = f'{river}_{year}_1_polygon'
    opath = os.path.join(oroot, oname)

    if not os.path.exists(oroot):
        os.makedirs(oroot)

    # Save polygon to disc
    with open(opath, "wb") as poly_file:
        pickle.dump(river_polygon, poly_file, pickle.HIGHEST_PROTOCOL)

# Load polygon from disc
# with open('tests/polygons/test_polygon', "rb") as poly_file:
#     loaded_polygon = pickle.load(poly_file)
