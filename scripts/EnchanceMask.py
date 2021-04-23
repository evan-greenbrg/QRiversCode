import glob
import os

from matplotlib import pyplot as plt
import numpy
import rasterio

from PyRivers.RasterHelpers import enhanceMask


date_range = [i for i in range(1986, 1988, 2)]
date_range = [2018]
root = '/Users/greenberg/Documents/PHD/Projects/BarT/riverData/brazos/'
for year in date_range:
    print(year)
    # Set up file paths
    imagefn = f'image/{year}/idx1/brazos_{year}_1_image.tif'
    maskfn = f'raw/{year}/idx1/brazos_{year}_1.tif'

    imagepath = os.path.join(root, imagefn)
    maskpath = os.path.join(root, maskfn)

    # Load raster datasets
    imageds = rasterio.open(imagepath)
    maskds = rasterio.open(maskpath)

    # Load the array objects
    image = imageds.read()
    mask = maskds.read()

    # Compute combined mask
    combined_mask = enhanceMask(image, mask).transpose().astype('uint8')


    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(mask[0,:,:])
    axs[1].imshow(combined_mask[0,:,:])
    plt.show()

    # Write to file
    outfn = f'raw/{year}/idx1/brazos_{year}_1_combined.tif'
    outpath = os.path.join(root, outfn)
    meta = maskds.meta.copy()
    with rasterio.open(outpath, "w", **meta) as dest:
        dest.write(combined_mask)

out = '/Users/greenberg/Documents/PHD/Projects/BarT/Figures/Methods/messyMask.tif'
meta = maskds.meta.copy()
with rasterio.open(out, "w", **meta) as dest:
    dest.write(combined_mask)
