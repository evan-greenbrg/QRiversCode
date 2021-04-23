import glob
import os

from matplotlib import pyplot as plt
import rasterio
from PyRivers import RasterHelpers 


river = 'beni'
year = 1985
stage = 'clipped'

for year in range(1985, 2001):
    # rootdir = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/raw/{year}'
    rootdir = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/{stage}/{year}'
    search_c = f'*/*{year}*.tif'
    q = os.path.join(rootdir, search_c)
    fps = glob.glob(q)

    out = f'{river}_{year}.tif'
    outpath = os.path.join(rootdir, out)

    RasterHelpers.files_to_mosaic(fps, outpath)
