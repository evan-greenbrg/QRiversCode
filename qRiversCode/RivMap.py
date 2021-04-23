import math
import json
import os

from affine import Affine
import cv2
import pandas
from shapely.geometry import box
from skimage import measure
import geopandas as gpd
from fiona.crs import from_epsg
from pycrs import parse as crsparse
import numpy
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask


def crop_to_mask(ds):
    """
    Crops an input image to its extents (i.e. removes columns/rows containing
    all zeros) 
    
    INPUTS:      image - Image data array 
                 meta - geotiff meta data
    
    OUTPUTS:     out_img - cropped image array 
                 meta - updated geotiff meta 
    """
    # Read mask array
    Icrop = ds.read(1)
    meta = ds.meta.copy()
    
    # Find x, y coordinates of mask boundaries
    cnts, hierarchy = cv2.findContours(
        Icrop.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Use the longest element
    olength = 0
    for i, cnt in enumerate(cnts):
        length = len(cnt)
        if length > olength:
            use = i
            olength = length

    # pick out the boundaries of object
    boundaries = cnts[use][:,0,:]
    x = boundaries[:, 0]
    y = boundaries[:, 1]

    # Find row/col positions of maximum and minimum mask values
    yBottom = math.ceil(max(y))
    xRight = math.ceil(max(x))
    yTop= math.ceil(min(y))
    xLeft = math.ceil(min(x))

    # Find coordinates of these positions
    left, top = rasterio.transform.xy(meta['transform'], yTop, xLeft)
    right, bot = rasterio.transform.xy(meta['transform'], yBottom, xRight)

    # Make geometry for mask
    bbox = box(right, bot, left, top)

    # Load into geopandas object
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])

    # Get coord object
    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    # Mask out the empty part of image
    out_img, out_transform = mask(
        dataset=ds,
        shapes=coords,
        crop=True
    )

    # Add two pixel boarder
#    height = out_img[0].shape[0]
#    out_img = out_img[0]
#    out_img = numpy.hstack(
#        [
#            numpy.zeros((height, 2)), 
#            out_img,
#            numpy.zeros((height, 2)), 
#        ]
#    )
#
#    width = out_img.shape[1]
#    out_img = numpy.vstack(
#        [
#            numpy.zeros((2, width)), 
#            out_img,
#            numpy.zeros((2, width)), 
#        ]
#    )

    meta.update({
        'transform': out_transform,
        'width': out_img.shape[2],
        'height': out_img.shape[1],
    })

    return out_img[0, :, :], meta 


def add_buffer(image, meta):
    """
    Adds 1 pixel buffer around the image file. Then updates that affine
    transformation in the meta data
    
    INPUTS:      image - Image data array 
                 meta - geotiff meta data
    
    OUTPUTS:     out_img - cropped image array 
                 meta - updated geotiff meta 
    """

    # Reduce dimension of image
    image = image[0, :, :]

    # row stack 
    h, w = image.shape 
    row_buffer = numpy.zeros((1, w), numpy.uint8)
    image = numpy.vstack((row_buffer, image, row_buffer))

    # column stack
    h, w = image.shape 
    col_buffer = numpy.zeros((h, 1), numpy.uint8)
    image = numpy.hstack((col_buffer, image, col_buffer))

    # Update affine transformation
    affine = [a for a in meta['transform']][0:-3]

    # shift x coord west
    affine[2] = affine[2] - affine[0]

    # shift y coord north 
    affine[5] = affine[5] - affine[4]

    # Update meta data
    meta['transform'] = Affine.from_gdal(*tuple(affine))

    # Reshape
    image = image[None, ...]

    return image, meta


def fill_holes(image, meta):
    """
    Fills any holes within the data array (within the channel banks).
    You do ahve to make sure the binary channel mask is 
    completely surrounded by 0 pixels
    
    INPUTS:      image - Image data array 
                 meta - geotiff meta data
    
    OUTPUTS:     out_img - cropped image array 
                 meta - updated geotiff meta 
    """
    # Reduce dimension of image
    image = image[0, :, :]
    
    # Copy to another variable
    image_flood = image.copy()

    # Get image shape
    h, w = image.shape

    # Initialize mask
    flood_mask = numpy.zeros((h+2, w+2), numpy.uint8)

    # Floodfill
    cv2.floodFill(image_flood, flood_mask, (0, 0), 255)

    # Invert floodfilled image
    image_flood_inv = cv2.bitwise_not(image_flood)

    # Combine two images to get foreground
    image_fill = image | image_flood_inv

    # Reshape
    image = image_fill[None, ...]

    return image, meta


def centerline_from_mask(image, meta):
    """
    Finds centerline from mask. Could implement this if I'm feeling bold at
    some point

    NOT DONE
    
    INPUTS:      image - Image data array 
                 meta - geotiff meta data
    
    OUTPUTS:     out_img - cropped image array 
                 meta - updated geotiff meta 
    """

    # Reduce dimension of image
    image = image[0, :, :]

    # Get image shape
    h, w = image.shape

    # Remove any spurious small patches by only keeping the largest area
    labels = measure.label(image)
    props = pandas.DataFrame(measure.regionprops_table(
        labels, 
        properties=(
            'area',
            'coords',
        )
    ))
    largest_prop = props[props['area'] == numpy.max(props['area'])]
    largest_coords = list(largest_prop['coords'])

    image = numpy.zeros(image.shape)
    for coord in largest_coords[0]:
        image[coord[0], coord[1]] = 1

    return image[0, :, :], meta


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    root = '/home/greenberg/ExtraSpace/PhD/Projects/BarT/beni/1985'
    fn = 'beni_1985_clean1.tif'
    fn_out = 'beni_1985_clean1_clip.tif'
    path = os.path.join(root, fn)
    opath = os.path.join(root, fn_out)

    ds = rasterio.open(path)
    image = ds.read(1)
    meta = ds.meta.copy()

    image, meta = crop_to_mask(image, meta)
    image, meta = add_buffer(image, meta)
    image, meta = fill_holes(image, meta)

    plt.imshow(image[0, :, :])
    plt.show()

