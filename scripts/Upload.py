import glob
import os
from google.cloud import storage

from PyRivers import Downloaders


bucket_name = 'earth-engine-rivmap'
river = 'beni'
stage = 'clipped'

for year in range(1985, 2001):
    print(year)

#    root = f'/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData/{river}/{stage}/{year}/*/*.tif'
    root = f'/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/{river}/{stage}/{year}/*/*.tif'
    fps = glob.glob(root)

    for fp in fps:
        idx = fp.split('/')[-2]
        Downloaders.pushRiverFiles(fp, bucket_name, river, stage, year, idx)
