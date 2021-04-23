import os
import timeit
import math
import copy

import pandas
import scipy
from scipy import spatial
from skimage import measure, draw, morphology, feature, graph
from skimage.morphology import medial_axis, skeletonize, thin, binary_closing
import numpy as np
import rasterio
from matplotlib import pyplot as plt


def getLargest(mask):
    labels = measure.label(mask)
     # assume at least 1 CC
    assert( labels.max() != 0 )

    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 
    cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return cc


def getCenterline(mask):
    labels = measure.label(mask)
     # assume at least 1 CC
    assert( labels.max() != 0 )

    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 
    cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    # Find skeletonized centerline
#    cc = morphology.binary_closing(cc)
#    cc = morphology.binary_erosion(cc)
#    cc = morphology.binary_dilation(cc)
    skeleton = skeletonize(cc, method='lee')
#    skeleton = medial_axis(cc)
    skeleton = thin(skeleton)

    return skeleton 


def findAllIntersections(centerline):
    rr, cc = np.where(centerline)

    rows = []
    cols = []
    # Get neighboring pixels
    for r, c in zip(rr, cc):
        window = centerline[r-1:r+2, c-1:c+2]

        if len(window[window]) > 3:
            rows.append(r)
            cols.append(c)

    return np.array([cols, rows]).transpose()


def findAllEndpoints(centerline):
    rr, cc = np.where(centerline)

    rows = []
    cols = []
    # Get neighboring pixels
    for r, c in zip(rr, cc):
        window = centerline[r-1:r+2, c-1:c+2]

        if len(window[window]) < 3:
            rows.append(r)
            cols.append(c)

    return np.array([cols, rows]).transpose()


def findRiverEndpoints(endpoints, es):
    es = [v for v in es] 
    riv_end = np.empty([2,2])
    for idx, v in enumerate(es):
        if v == 'N':
            i = np.where(endpoints[:,1] == endpoints[:,1].min())[0][0]
        elif v == 'E':
            i = np.where(endpoints[:,0] == endpoints[:,0].max())[0][0]
        elif v == 'S':
            i = np.where(endpoints[:,1] == endpoints[:,1].max())[0][0]
        elif v == 'W':
            i = np.where(endpoints[:,0] == endpoints[:,0].min())[0][0]
        riv_end[idx, :] = endpoints[i,:]

    return riv_end


def fillHoles(mask, thresh=40):
    # Find contours
    contours = measure.find_contours(mask, 0.8)
    # Display the image and plot all contours found
    for contour in contours:
        # Get polygon
        poly = draw.polygon(contour[:, 0], contour[:, 1])
        area = (
            (poly[0].max() - poly[0].min() + 1)
            * (poly[1].max() - poly[1].min() + 1)
        )
        # Filter by size
        if area < thresh:
            draw.set_color(
                mask, 
                poly, 
                True 
            )

    return mask


def removeSmallSegments(centerline, intersections, endpoints, thresh):
    tree = spatial.KDTree(intersections)
    costs = np.where(centerline, 1, 1000)
    removed = 0
    for point in endpoints:
        distance, i = tree.query(point)
        path, dist = graph.route_through_array(
            costs, 
            start=(point[1], point[0]),
            end=(intersections[i][1], intersections[i][0]),
            fully_connected=True
        )

        path = np.array(path)
        if dist < thresh:
            centerline[path[:,0], path[:,1]] = False
            removed += 1
        else:
            continue
        
    centerline[intersections[:,1], intersections[:,0]] = True

    return centerline, removed


def cleanCenterline(centerline, es, thresh=10000):
    removed = 999
    centerline = getLargest(centerline)
    endpoints = findAllEndpoints(centerline)
    # Find the terminal endpoints
    river_endpoints = findRiverEndpoints(endpoints, es)
    while removed > 2:
        # Find the all endpoints in the centerline
        endpoints = findAllEndpoints(centerline)
        # Add an intersection
        for end in river_endpoints:
            centerline[
                int(end[1]-1):int(end[1]+2),
                int(end[0]
            )] = 1
            centerline[
                int(end[1]), 
                int(end[0]-1):int(end[0]+2)
            ] = 1

        # Find all intersections
        intersections = findAllIntersections(centerline)

        # Remove all the small bits
        centerline, removed = removeSmallSegments(
            centerline,
            intersections, 
            endpoints,
            thresh
        )
        print(removed)

    # Remove the fake intersection created at the river ends
    for end in river_endpoints:
        centerline[
            int(end[1]-1):int(end[1]+2),
            int(end[0]
        )] = 0
        centerline[
            int(end[1]), 
            int(end[0]-1):int(end[0]+2)
        ] = 0

    return centerline, river_endpoints


def getCenterlineExtent(centerline, river_endpoints, maxdistance=500):
    centerline_i = np.array(
        np.where(centerline == True)
    ).transpose()
    tree = spatial.KDTree(centerline_i)

    total = []
    for endpoint in river_endpoints:
        distance, neighbor = tree.query([endpoint[1], endpoint[0]])
        if distance > maxdistance:
            total.append(False)
        else:
            total.append(True)

    if False in total:
        return False
    else:
        return True


if __name__=='__main__':

    root = 'tests/classification/'
    name = 'classified_beni_1986.tif'
    ipath = os.path.join(root, name)

    ds = rasterio.open(ipath)
    image = ds.read()
    image = image[0, :, :]

    image = fillHoles(image)
    centerline = getCenterline(image)
    centerline = cleanCenterline(centerline)

    fig, axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=(8, 4),
        sharex=True, 
        sharey=True
    )

    axes[0].imshow(image)
    axes[1].imshow(centerline)
    axes[1].scatter(endpoints[:,0], endpoints[:,1])
    axes[1].scatter(intersections[:,0], intersections[:,1])
    plt.show()



#    riv_endpoints = findRiverEndpoints(endpoints, 'NS')
#
#
#    costs = np.where(centerline, 1, 1000)
#    path, cost = graph.route_through_array(
#        costs, 
#        start=(int(riv_endpoints[0][1]), int(riv_endpoints[0][0])),
#        end=(int(riv_endpoints[1][1]), int(riv_endpoints[1][0])),
#        fully_connected=True
#    )
#    path = np.array(path)

