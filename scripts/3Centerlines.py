import os
import re
import glob
import math

import pandas
import rasterio
from rasterio import transform
from matplotlib import pyplot as plt


def getDataFiles(data_fps):
    """"
    From list of files in directory structure, return data files by type
    """
    data_files = {
        'centerline': [],
        'width': [],
        'curvature': []
    }
    for data_fp in data_fps:
        if re.match(r".*/.*centerline.csv", data_fp):
            data_files['centerline'].append(data_fp)
        if re.match(r".*/.*widths.csv", data_fp):
            data_files['width'].append(data_fp)
        if re.match(r".*/.*curvatures.csv", data_fp):
            data_files['curvature'].append(data_fp)

    return data_files


def buildDataFrame(ds, centerline, width, curvature):
    # Save image data to dataframe
    data = {
        'longitude': [],
        'latitude': [],
        'width': [],
        'curvature': [],
    }
    for idx, row in centerline.iterrows():
        lon, lat = ds.xy(row['row'], row['col'])

        data['longitude'].append(lon)
        data['latitude'].append(lat)
        data['width'].append(width.iloc[idx][0] * PIXEL_SIZE)
        data['curvature'].append(curvature.iloc[idx][0] * PIXEL_SIZE)

    return pandas.DataFrame(data)


def stackAllIdxs(year_dfs):
    """
    Takes the list of dataframes and turns it into a single dataframe
    """
    if len(year_dfs) == 0:
        return year_dfs[0]
    else:
        return pandas.concat(year_dfs).reset_index(drop=True)


PIXEL_SIZE = 30     # Landsat 30m pixels
river = 'ica'
save_idx = True

for year in range(2004, 2018):
    # image_root = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/clipped/{year}/*/*.tif'
    image_root = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/clipped/{year}/*/*.tif'

    # data_root = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/data/{year}/*/*.csv'
    data_root = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/data/{year}/*/*.csv'

    # Get image files
    image_fps = glob.glob(image_root)

    # Get data files
    data_fps = glob.glob(data_root)
    data_files = getDataFiles(data_fps)

    if len(data_fps) == 0:
        continue
    else:
        print(year)

    year_dfs = []
    for idx, image_fp in enumerate(image_fps):
        print(idx)

        # Load the image
        ds = rasterio.open(image_fp)

        # Load the centerline 
        centerline = pandas.read_csv(
            data_files['centerline'][idx]
            , names=['col', 'row']
        )

        # Load the width
        width = pandas.read_csv(
            data_files['width'][idx], 
            names=['width']
        )

        # Load the curvature 
        curvature = pandas.read_csv(
            data_files['curvature'][idx], 
            names=['curvature']
        )
        curvature = curvature.drop([0, 1]).reset_index(drop=True)

        # Find Spacing and resample at width spacing
        spacing = int(round((len(centerline) / len(width)), 0))
        centerline = centerline.iloc[::spacing].reset_index(drop=True)
        curvature = curvature.iloc[::spacing].reset_index(drop=True)

        # Make sure these two dimensions match
        if len(centerline) > len(width):
            centerline = centerline.iloc[:len(width)]
        if len(centerline) > len(curvature):
            centerline = centerline.iloc[:len(curvature)]

        # Build dataframes and append all idxs
        year_df = buildDataFrame(ds, centerline, width, curvature)
        year_dfs.append(year_df)

        if save_idx:
            outpath = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/data/{year}/idx{idx+1}/{river}_{year}_data.csv'
            year_df.to_csv(outpath)

    if not save_idx:
        # Stack the list of dataframes
        data_df = stackAllIdxs(year_dfs)

        # outpath = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/data/{year}/{river}_{year}_data.csv'
        outpath = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/data/{year}/{river}_{year}_data.csv'
        data_df.to_csv(outpath)
