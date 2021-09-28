import os
import math

import pandas
from scipy import spatial
from scipy.signal import savgol_filter
import networkx as nx
import numpy as np
from osgeo import ogr
from osgeo import osr
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from qRiversCode.GraphSort import getGraph, GraphSort


def getNeighboringPoints(G, node, n=5):
    # How many edges is the node attached to?
    node_edges = list(G.edges(node))

    close_nodes = []
    # If the node has no edges, return None
    if not len(node_edges):
        return []

    # If the node only has one edge
    elif len(node_edges) == 1:
        close_nodes = nx.ego_graph(
            G,
            node,
            radius=n-1,
            center=True,
            undirected=False,
            distance=None
        )

    else:
        close_nodes = nx.ego_graph(
            G,
            node,
            radius=n-2,
            center=True,
            undirected=False,
            distance=None
        )

    return close_nodes


def calculateCurvature(points, node):

    s = points[points['idx'] == node]['distance']
    ss = np.array(points['distance'])
    xs = np.array(points['easting'])
    ys = np.array(points['northing'])

    # Fit Poly x
    try:
        np.warnings.filterwarnings('ignore')
        xz = np.polyfit(ss, xs, deg=3)
        px = np.poly1d(xz)
        xz1 = np.polyder(px, m=1)
        xz2 = np.polyder(px, m=2)

        # Fit Poly y
        yz = np.polyfit(ss, ys, deg=3)
        py = np.poly1d(yz)
        yz1 = np.polyder(py, m=1)
        yz2 = np.polyder(py, m=2)

        # Get derivatives
        dxds = xz1(s)
        d2xds2 = xz2(s)

        dyds = yz1(s)
        d2yds2 = yz2(s)

        # Get Curvature
        C = (
            ((dxds * d2yds2) - (dyds * d2xds2))
            / (((dxds**2) + (dyds**2))**(3/2))
        )
    except:
        C = None

    return C


def findEPSG(longitude, latitude):
    zone = math.ceil((longitude + 180) / 6)

    if latitude > 0:
        espg_format = '326{}'
    else:
        espg_format = '327{}'

    if zone < 10:
        zone = '0' + str(zone)

    return espg_format.format(str(zone))


def transform_coordinates(pointX, pointY, iEPSG, oEPSG):
    """
    Transforms set of coordinates from one coordinate system to another
    """
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pointX, pointY)

    # create coordinate transformation
    source = osr.SpatialReference()
    source.ImportFromEPSG(iEPSG)

    target = osr.SpatialReference()
    target.ImportFromEPSG(oEPSG)

    coordTransform = osr.CoordinateTransformation(
        source,
       target 
    )

    # transform point
    point.Transform(coordTransform)

    return point.GetX(), point.GetY()


def getCurvature(G, centerline):

    curvatures = []
    for node in G.nodes:
        # Get 3 closest nodes
        node_path = getNeighboringPoints(G, node)
        source = min(list(node_path.nodes))
        target = max(list(node_path.nodes))
        path = nx.shortest_path(
            node_path, 
            weight='length', 
            source=source,
            target=target
        )
        points = centerline.iloc[path]
        points = centerline.iloc[path]
        points['idx'] = points.index
        points = points.reset_index(drop=True)

        # convert to UTM
        eastings = []
        northings = []
        for idx, point in points.iterrows():
            epsg = findEPSG(point['x'], point['y'])
            x, y = transform_coordinates(
                point['y'],
                point['x'],
                4326,
                int(epsg)
            )
            eastings.append(x)
            northings.append(y)

        points['easting'] = eastings
        points['northing'] = northings

        # get distance
        distance = [0]
        dist = 0
        for idx, row in points.iterrows():
            if idx == 0:
                continue
            a = np.array([
                points.iloc[idx-1]['easting'], 
                points.iloc[idx-1]['northing']
            ])
            b = np.array([row['easting'], row['northing']])
            dist += np.linalg.norm(a-b)
            distance.append(dist)
        points['distance'] = distance

        # Calculate Curvature
        C = calculateCurvature(points, node)
        if C:
            curvatures.append(C[0])
        else:
            curvatures.append(None)

    return curvatures

