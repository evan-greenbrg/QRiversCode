import os
from google.cloud import storage

from PyRivers import Downloaders


# outroot = '/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData'
#outroot = '/home/greenberg/ExtraSpace/PhD/Projects/BarT/riverData/'
outroot = '/Users/greenberg/Documents/PHD/Projects/Chapter2/data/'
bucket_name = 'earth-engine-rivmap'
river = 'Chehalis'
stage = 'image'
years = [1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2013, 2016, 2020]
# years = [2012, 2016, 2020]
# years = [2013]

for year in years:
# for year in years:
    Downloaders.pullRiverFiles(outroot, bucket_name, river, year, stage)
