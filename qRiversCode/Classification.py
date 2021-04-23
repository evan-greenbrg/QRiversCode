import os
import timeit
import math

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
import rasterio
import geopandas as gpd
from rasterio import plot
from rasterio.mask import mask
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


class pickData(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.events = []
        self.points = []
        self.rects = []

    def clear(self, event):
        # Clear the most recent box of pointd
        self.events = []
        self.X0 = None

        # Remove all plotted picked points
        self.rect.remove()
        for p in self.points:
            if p:
                p.remove()

        # Remove most recent rectangle
        self.rects.pop(-1)
        self.points = []
        self.rects = []
        self.rect = None
        print('Cleared')

    def done(self, event):
        # All done picking points
        plt.close('all')
        print('All Done')

    def draw_box(self, event):
        # Draw the points box onto the image
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
        self.rects.append(r)

        for p in self.points:
            p.remove()
        self.points = []

        event.canvas.draw()

    def __call__(self, event):
        # Call the events
        self.event = event

        if not event.dblclick:
            return 0

        self.x, self.y = event.xdata, event.ydata 
        self.events.append((self.x, self.y))
        self.events = list(set(self.events))

        if self.x is not None:
            # Plot where the picked point is
            self.points.append(self.ax.scatter(self.x, self.y))
            event.canvas.draw()

        if len(self.events) == 2:
            self.draw_box(event)
            self.events = []


def dataBox2df(sample, bandnames):
    height = sample.shape[1] * sample.shape[2]
    matrix = np.zeros([height, sample.shape[0]])
    n = 0
    for i in range(sample.shape[1]):
        for j in range(sample.shape[2]):
            matrix[n, :] = sample[:, i, j]
            n +=1

    return pandas.DataFrame(matrix, columns=bandnames)


def generateTrainingData(image, surface_class, bandnames):
    im = image[:, :, :].transpose()

    ims = np.zeros([image.shape[1], image.shape[2], 3])
    for i, band in enumerate([6, 4, 2]):
        ims[:,:,i] = plot.adjust_band(image[band,:,:], kind='linear')

    # Get Water data
    fig = plt.figure()
    t = plt.gca()
    im = plt.imshow(ims)

    PD = pickData(t)

    axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
    bclear = Button(plt.gca(), 'Clear')
    bclear.on_clicked(PD.clear)

    axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
    bdone = Button(plt.gca(), 'Done')
    bdone.on_clicked(PD.done)

    fig.canvas.mpl_connect('button_press_event', PD)

    im.set_picker(5) # Tolerance in points

    plt.show()

    # Convert rectangles to DF 
    df = pandas.DataFrame()
    for rect in PD.rects:
        # Get indexes at bottom left
        botleft = rect.get_xy()
        botleft = [math.ceil(i) for i in botleft]

        # Get indexes at top right
        topright = [
            botleft[0] + rect.get_width(),
            botleft[1] + rect.get_height(),
        ]
        topright = [math.ceil(i) for i in topright]

        # Get image rows
        ys = [botleft[1], topright[1]]
        xs = [botleft[0], topright[0]]
        sample = image[:, min(ys):max(ys),min(xs):max(xs)]
        print(sample.shape)

        df = pandas.concat(
            [df, dataBox2df(sample, bandnames)]
        ).reset_index(drop=True)

    df['class'] = surface_class 

    return df


def generateTreeQ(raster, water, land):
    # Load in the raster
    ds = rasterio.open(raster.source())
    bandnames = ds.descriptions
    image = ds.read()

    # Generate water points
    water = gpd.read_file(water.source())
    out, _ = mask(ds, water.geometry, invert=False, filled=False)
    mask_ar = np.invert(out.mask)
    data = out.data

    band_data = {name: [] for name in bandnames}
    for i, name in enumerate(band_data.keys()):
        band_data[name] = data[i][mask_ar[i]]

    water_df = pandas.DataFrame(band_data)
    water_df['class'] = [1 for i in range(len(water_df))]

    # Generate land points
    land = gpd.read_file(land.source())
    out, _ = mask(ds, land.geometry, invert=False, filled=False)
    mask_ar = np.invert(out.mask)
    data = out.data

    band_data = {name: [] for name in bandnames}
    for i, name in enumerate(band_data.keys()):
        band_data[name] = data[i][mask_ar[i]]

    not_water_df = pandas.DataFrame(band_data)
    not_water_df['class'] = [0 for i in range(len(not_water_df))]

    # Set up whole df
    df = pandas.concat([water_df, not_water_df])

    # Remove Nan
    df = df.dropna(how='any')

    # Initialize tree
    clf = DecisionTreeClassifier(
        random_state=0, 
        max_depth=5
    )

    feature_cols = [b for b in bandnames]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols], 
        df['class'], 
        test_size=0.1, 
        random_state=1
    )

    clf = clf.fit(
        x_train,
        y_train
    )

    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf



def generateTree(inpath):
    """
    Produces a decision tree classifier that can be used 
    across multiple images
    """
    # Load in the image
    ds = rasterio.open(inpath)
    bandnames = ds.descriptions
    image = ds.read()

    # Water
    print('Pick Water Points')
    water_df = generateTrainingData(image, 1, bandnames)

    # Not water
    print('Pick Non-Water Points')
    not_water_df = generateTrainingData(image, 0, bandnames)

    # Set up whole df
    df = pandas.concat([water_df, not_water_df])

    # Remove Nan
    df = df.dropna(how='any')
    print(df.head())

    # Initialize tree
    clf = DecisionTreeClassifier(
        random_state=0, 
        max_depth=5
    )

    feature_cols = [b for b in bandnames]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols], 
        df['class'], 
        test_size=0.1, 
        random_state=1
    )

    clf = clf.fit(
        x_train,
        y_train
    )

    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf


def predictPixelsQ(ds, clf):
    bandnames = ds.descriptions
    image = ds.read()

    # Reshape to correct shape
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])
    image_predict = np.moveaxis(image, 0, -1)
    img_as_array = image_predict[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(
        o=image.shape,
        n=img_as_array.shape)
    )

    # Crazy method to predict for each pixel
    predictions = np.empty([img_as_array.shape[0],])
    predictions[:] = None
    for i, row in enumerate(img_as_array):
        if len(row[~np.isnan(row)]) > 0:
            predictions[i] = clf.predict(row.reshape(1, len(bandnames)))[0]

    # Reshape our classification map
    class_prediction = predictions.reshape(image_predict[:, :, 0].shape)

    return class_prediction.astype(rasterio.int8)


def predictPixels(inpath, opath, clf):

    ds = rasterio.open(inpath)
    bandnames = ds.descriptions
    image = ds.read()

    # Reshape to correct shape
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])
    image_predict = np.moveaxis(image, 0, -1)
    img_as_array = image_predict[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(
        o=image.shape,
        n=img_as_array.shape)
    )

    # Crazy method to predict for each pixel
    predictions = np.empty([img_as_array.shape[0],])
    predictions[:] = None
    for i, row in enumerate(img_as_array):
        if len(row[~np.isnan(row)]) > 0:
            predictions[i] = clf.predict(row.reshape(1, len(bandnames)))[0]

    # Reshape our classification map
    class_prediction = predictions.reshape(image_predict[:, :, 0].shape)
    print(class_prediction.shape)
#    class_prediction = class_prediction[0, :, :]

    # Reshape our classification map
#    class_prediction = class_prediction.reshape(img[:, :, 0].shape)

    # Output Class Predictions
    meta = ds.meta.copy()
    meta.update({'dtype': rasterio.int8, 'count': 1})
    with rasterio.open(opath, "w", **meta) as dest:
        dest.write(class_prediction.astype(rasterio.int8), 1)

    return class_prediction 
