import glob
import os

from matplotlib import pyplot as plt
import rasterio
from PyRivers import RivMap 

river = 'ica'

i = 1
for year in range(2004, 2022, 2):
#    rootdir = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/clean/{year}/*'
    rootdir = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/clean/{year}/idx{i}'
    # rootdir = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/clean/{year}/*/'

    search_c = f'*{year}*clean*.tif'

    q = os.path.join(rootdir, search_c)
    fps = glob.glob(q)

    if len(fps) == 0:
        continue

    for fp in fps:

    #    rootdir = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/clipped/{year}/idx{i}'
#        rootdir = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/clipped/{year}/idx{i}'
        rootdir = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/clipped/{year}/idx{i}'

        fn_out = f'{river}_{year}__clip.tif'
        opath = os.path.join(rootdir, fn_out)
        print(opath)

        ds = rasterio.open(fp)
        image = ds.read()
        meta = ds.meta.copy()

        image, meta = RivMap.crop_to_mask(ds)

        if not os.path.exists(rootdir):
            os.makedirs(rootdir)

        with rasterio.open(opath, "w", **meta) as dest:
            dest.write(image.astype(rasterio.uint8), 1)

