import glob
import re
import os
import joblib
from PyRivers import Classification

pattern = '(.*)\/(\w*)\/.*\/(\d{4})\/(\w*)\/'

river = 'Chehalis'
root = f'/Users/greenberg/Documents/PHD/Projects/Chapter2/Data/{river}/**'
inname = f'{river}*.tif'
inpath = os.path.join(root, inname)
fps = glob.glob(inpath, recursive=True)

for i, fp in enumerate(fps):
    regex = re.search(pattern, fp)
    root = regex.group(1)
    river = regex.group(2)
    year = regex.group(3)
    idx = regex.group(4)

    if i == 0:
        clf = Classification.generateTree(fp)
        clfpath = os.path.join(
            root,
            river,
            f'{river}_clf.joblib.pkl'
        )
        joblib.dump(clf, clfpath, compress=9)

    oroot = os.path.join(
        root,
        river,
        'mask',
        year,
        idx
    )
    oname = f'{river}_{year}_1_mask.tif'
    opath = os.path.join(oroot, oname)

    if not os.path.exists(oroot):
        os.makedirs(oroot)
        
    prediction = Classification.predictPixels(fp, opath, clf)
