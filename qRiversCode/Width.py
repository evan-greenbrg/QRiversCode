import os
import math

import rasterio
from rasterio import transform
from skimage import measure
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy import spatial
from matplotlib import pyplot as plt
import numpy as np
import pandas
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point 
from shapely import affinity
from shapely import ops
import geopandas as gpd
import networkx as nx

from qRiversCode import Centerline 


def cleanChannel(image):
    labels = measure.label(image)
     # assume at least 1 CC
    assert( labels.max() != 0 )
    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 
    channel = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    channel = Centerline.fillHoles(channel)

    return channel


def getDirection(centerline_i, neighbors):

    # Get direction of the centerline at segment
    # Get lowest index
    mini = centerline_i[neighbors.min()]
    # Get largest index
    maxi = centerline_i[neighbors.max()]
    # Get x change
    dx = (mini[0] - maxi[0])
    # Get y change
    dy = (mini[1] - maxi[1])

    # Get direction of cross-section at segment
    cross_dx = -dy
    cross_dy = dx

    return cross_dx, cross_dy


def createCrossSection(location, direction, xprop, yprop):
    xoffset = direction[0] * xprop
    yoffset = direction[1] * yprop

    A = [(location[0] + xoffset), (location[1] + yoffset)]
    B = [(location[0] - xoffset), (location[1] - yoffset)]

    return LineString([A, B])


def createChannelPolygon(contours):
    polygons = []
    longest = 0
    # Make polygons and find the longest contour
    for i, contour in enumerate(contours):

        # Find the longest polygon
        if len(contour) > longest:
            longest_i = i
            longest = len(contour)

        # Append to list of polygons
        polygons.append(Polygon(contour))

    # find which polygons fall within the longest
    inners = []
    for i, polygon in enumerate(polygons):
        # Save the river polygon
        river_polygon = polygons[longest_i]

        # If the polygon is the river polygon move to the next
        if i == longest_i:
            continue

        # See if the polygon is within the river polygons
        if river_polygon.contains(polygon):
            inners.append(polygon)

    return Polygon(
        river_polygon.exterior.coords, 
        [inner.exterior.coords for inner in inners]
    )


def intersectionWidth(segment, cross_section, river_poly):
    # Find coordinates where there is intersection

    # Have to handle the error if there are multiple points
    inter = cross_section.intersection(river_poly.buffer(0))
    if inter.geom_type == 'LineString':
        if len(inter.coords) < 2:
            return None, np.empty(0)
    try:
        intersect_coords = np.array(
            inter.xy
        ).transpose()
    except NotImplementedError:
        inters = cross_section.intersection(river_poly.buffer(0))
        if len(inters) == 0:
            return None, np.empty(0)

        for i, inter in enumerate(inters):
            if i == 0:
                intersect_coords = np.array(inter.xy).transpose()
            else:
                intersect_coords = np.vstack([
                    intersect_coords,
                    np.array(inter.xy).transpose()
                ])

#    t1 = river_poly.buffer(0).exterior.xy
#    t2 = cross_section.xy
#    plt.scatter(t1[0], t1[1])
#    plt.plot(t2[0], t2[1])
#    for inter in river_poly.interiors:
#        t3 = inter.xy
#        plt.scatter(t3[0], t3[1])
#    plt.show()

    # Find distances between segment and intersecting points
    tree = spatial.KDTree(intersect_coords)
    distance, neighbors = tree.query(segment, 2)

    if len(intersect_coords) < 2:
        return None, np.empty(0)
    width_points = intersect_coords[neighbors]

    return np.linalg.norm(width_points[0]-width_points[1]), width_points


def sortCenterline(centerline_i):
    """
    This method unfortunately reduces the centerline to a single path
    """
    G = nx.Graph()
    tree = KDTree(centerline_i, leaf_size=2, metric='euclidean')  # Create a distance tree
    for p in centerline_i:
        dist, ind = tree.query(p.reshape(1,-1), k=3)
        p = (p[0], p[1])
        G.add_node(p)

        n1, l1 = centerline_i[ind[0][1]], dist[0][1]
        n2, l2 = centerline_i[ind[0][2]], dist[0][2]

        n1 = (n1[0], n1[1])
        n2 = (n2[0], n2[1])
        G.add_edge(p, n1)
        G.add_edge(p, n2)

    source = tuple(centerline_i[0])
    target = tuple(centerline_i[-1])

    return np.array(
        nx.shortest_path(G, source=source, target=target)
    )


def getWidths(mask, centerline, step=5):
    channel = cleanChannel(mask)
    channel[:, 0] = 0
    channel[:, -1] = 0
    channel[0, :] = 0
    channel[-1, :] = 0
    contours = measure.find_contours(channel, 0.5, fully_connected='high')

    # Get centerline steps
    centerline_i = np.array(np.where(centerline == 1)).transpose()

    segments_i = centerline_i[0::step]
    # Initialize the tree
    tree = spatial.KDTree(centerline_i)
    # Convert the channel to polygon object
    river_poly = createChannelPolygon(contours)

    # Structure for widths
    data = {
        'rowi': [],
        'coli': [],
        'width': [],
        'width_rowi': [],
        'width_coli': [],
    }
    for i, segment in enumerate(segments_i):
        if i == 0:
            continue
        # Get direction of the channel in the segment
        distance, neighbors= tree.query(segment, 5)
        direction = getDirection(centerline_i, neighbors)
        cross_section = createCrossSection(segment, direction, 15, 15)
        width, width_points = intersectionWidth(segment, cross_section, river_poly)

        data['rowi'].append(segment[1])
        data['coli'].append(segment[0])
        data['width'].append(width)
        if len(width_points) == 0:
            data['width_rowi'].append(None)
            data['width_coli'].append(None)
        else:
            data['width_rowi'].append(width_points[:, 1])
            data['width_coli'].append(width_points[:, 0])

    width_df = pandas.DataFrame(data)

    return width_df, river_poly


def getCoordinates(dstransform, width_df):
    new_data = {
        'x': [],
        'y': [],
        'width_x': [],
        'width_y': []
    }
    for idx, row in width_df.iterrows():
        x, y = transform.xy(dstransform, row['coli'], row['rowi'])
        new_data['x'].append(x)
        new_data['y'].append(y)

        width_x, width_y = transform.xy(
            dstransform, 
            row['width_coli'], 
            row['width_rowi']
        )
        new_data['width_x'].append(width_x)
        new_data['width_y'].append(width_y)

    width_df['x'] = new_data['x']
    width_df['y'] = new_data['y']
    width_df['width_x'] = new_data['width_x']
    width_df['width_y'] = new_data['width_y']

    return width_df


def getWidth(centerlinepath, maskpath, step):
    # Load centerline image
    ds = rasterio.open(centerlinepath)
    centerline = ds.read(1)

    # Load mask image
    ids = rasterio.open(maskpath)
    mask = ids.read(1)

    # Get Width
    width_df, river_polygon = getWidths(
        mask, 
        centerline, 
        step=step
    )
    width_df = width_df.dropna(how='any')
    width_df = getCoordinates(ds.transform, width_df)

    return width_df, river_polygon


if __name__=='__main__':
    centerlinepath = centerline
    maskpath =  mask

    step = 1
