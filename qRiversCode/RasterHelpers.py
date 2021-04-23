import copy
import glob
import os

from skimage.measure import label   
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
from rasterio.merge import merge


class maskEraser(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax, ds):
        self.ax = ax
        self.events = []
        self.ds = ds 
        self.ds_new = copy.deepcopy(ds)
        self.points = []

    def clear(self, event):
        self.events = []
        self.X0 = None

        # Remove all plotted picked points
        self.rect.remove()
        for p in self.points:
            p.remove()
        self.points = []
        self.rect = None
        print('Cleared')

    def delete(self, event):
        firstcorn = [int(round(i)) for i in self.events[0]]
        secondcorn = [int(round(i)) for i in self.events[1]]

        xs = [firstcorn[1], secondcorn[1]]
        ys = [firstcorn[0], secondcorn[0]]

        self.ds_new[min(xs):max(xs), min(ys):max(ys)] = 0
        self.ax.imshow(self.ds_new)

        self.rect.remove()
        self.rect = None
        self.events = []

    def done(self, event):
        plt.close('all')
        print('All Done')

    def draw_box(self, event):
        width = self.events[1][0] - self.events[0][0]
        height = self.events[1][1] - self.events[0][1]
        r = Rectangle(
            self.events[0],
            width,
            height,
            color='red',
            fill=False
        )
        self.rect = self.ax.add_patch(r)

        for p in self.points:
            p.remove()
        self.points = []

        event.canvas.draw()

    def __call__(self, event):
        self.event = event

        if not event.dblclick:
            return 0 

        self.x, self.y = event.xdata, event.ydata 

        self.events.append((self.x, self.y))
        self.events = list(set(self.events))

        if len(self.events) > 2:
            return 0

        if self.x is not None:
            # Plot where the picked point is
            self.points.append(self.ax.scatter(self.x, self.y))
            event.canvas.draw()

        # Handle which event it was
        if len(self.events) == 2:
            self.draw_box(event)


def files_to_mosaic(fps, outpath, write=True):
    src_files_to_mosaic = []
    for fp in fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        }
    )

    if write:
        with rasterio.open(outpath, "w", **out_meta) as dest:
            dest.write(mosaic)

    return mosaic


def cleanRaster(fp, outpath):
    ds = rasterio.open(fp)
    ds_meta = ds.meta.copy()

    ds = ds.read(1)

    fig = plt.figure()
    t = plt.gca()
    im = plt.imshow(ds)
    DC = maskEraser(t, ds)

    axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
    bclear = Button(plt.gca(), 'Clear')
    bclear.on_clicked(DC.clear)

    axdelete = plt.axes([0.1, 0.0, 0.1, 0.1])
    bdelete = Button(plt.gca(), 'Delete')
    bdelete.on_clicked(DC.delete)

    axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
    bdone = Button(plt.gca(), 'Done')
    bdone.on_clicked(DC.done)

    fig.canvas.mpl_connect('button_press_event', DC)

    im.set_picker(2) # Tolerance in points

    plt.show()

    dsnew = DC.ds_new
    print(dsnew.shape)
    ds_meta.update(
        {
            "height": dsnew.shape[0],
            "width": dsnew.shape[1],
        }
    )

    if outpath:
        with rasterio.open(outpath, "w", **ds_meta) as dest:
            dest.write(dsnew[None,...])

    return dsnew, ds_meta


def WaterIndex(red, swir):
    """
    Discriminating index for identifying water
    """
    return (red - swir) / (red + swir)


def ImageConvert(image):
    new_image = np.zeros(image.shape)
    for idx, im in enumerate(image):
        # Because I don't know what the no data value is I have to do this
        # Setting up array where no data is 0
        new_im = np.zeros(im.shape)
        for i, row in enumerate(im):
            for j, val in enumerate(row):
                if val > 0:
                    new_im[i, j] = val

        # Get max in the band
        band_max = new_im.max()
        new_im = new_im / band_max

        # Save in the image
        new_image[idx] = new_im

    return new_image.transpose()


def getLargestCC(combined_mask):
    labels = label(combined_mask)
     # assume at least 1 CC
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def enhanceMask(image, mask):
    # Calculate index over image
    image = image.transpose()
    mask = mask.transpose()
    index_image = np.zeros((mask.shape[0], mask.shape[1]))
    for i, row in enumerate(image):
        for j, vals in enumerate(row):
            if vals[0] > 0:
                index_image[i, j] = WaterIndex(vals[3], vals[4])
            else:
                index_image[i, j] = 0


    # convert to binrary
    index_image[index_image < 0] = 0 
    index_image[index_image > 0] = 1 

    # Combine the two masks
    mask = mask[:,:,0]
    combined_mask = index_image + mask

    # convert to binrary
    combined_mask[combined_mask < 0] = 0 
    combined_mask[combined_mask > 0] = 1 

#    combined_mask = getLargestCC(combined_mask)

    return combined_mask.reshape((
        combined_mask.shape[0],
        combined_mask.shape[1],
        1
    ))


if __name__ == '__main__':
    rootdir = '/home/greenberg/ExtraSpace/PhD/Projects/BarT'
    search_c = 'beni*.tif'
    q = os.path.join(rootdir, search_c)
    fps = glob.glob(q)

    out = 'beni_2018.tif'
    outpath = os.path.join(rootdir, out)

    files_to_mosaic(fps, outpath)


    ds = rasterio.open(outpath)
    ds = ds.read(1)

    fig = plt.figure()
    t = plt.gca()
    im = plt.imshow(ds)
    DC = maskEraser(t, ds)

    axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
    bclear = Button(plt.gca(), 'Clear')
    bclear.on_clicked(DC.clear)

    axdelete = plt.axes([0.1, 0.0, 0.1, 0.1])
    bdelete = Button(plt.gca(), 'Delete')
    bdelete.on_clicked(DC.delete)

    axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
    bdone = Button(plt.gca(), 'Done')
    bdone.on_clicked(DC.done)

    fig.canvas.mpl_connect('button_press_event', DC)

    im.set_picker(5) # Tolerance in points

    plt.show()

    fp = outpath
    outpath = os.path.join(rootdir, 'beni_2018_clean.tif')
    dsnew, dsmeta = cleanRaster(fp, outpath)


