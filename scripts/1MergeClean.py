import glob
import os

from matplotlib import pyplot as plt
import rasterio
from PyRivers import RasterHelpers 

river = 'ica'
year = 2019
i = 1

for year in range(2004, 2022, 2):
    io = 1
    i = 1
    # rootdir = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/raw/{year}'
    # rootdir = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/raw/{year}'
    rootdir = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/raw/{year}/idx{i}'
    search_c = f'*{year}*_1.tif'
    q = os.path.join(rootdir, search_c)
    fps = glob.glob(q)
    if len(fps) == 0:
        continue

    out = f'{river}_{year}.tif'
    outpath = os.path.join(rootdir, out)

    RasterHelpers.files_to_mosaic(fps, outpath)

    out = f'{river}_{year}_clean.tif'
    # rootdir = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/clean/{year}/idx{i}'
    # rootdir = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/clean/{year}/idx{i}'
    rootdir = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/clean/{year}/idx{io}'
    clean_outpath = os.path.join(rootdir, out)

    # Check if path exists
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

    RasterHelpers.cleanRaster(outpath, clean_outpath)

