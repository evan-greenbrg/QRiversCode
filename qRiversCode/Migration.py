import os
from datetime import datetime
import math
from itertools import islice
from itertools import tee

import pickle
import rasterio
import pandas
from scipy import spatial
from scipy.signal import savgol_filter
import numpy
from numpy.matlib import repmat
import geopandas as gpd
import networkx as nx
from osgeo import ogr
from osgeo import osr
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from shapely import geometry
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point 
from shapely.geometry import MultiPoint 
from shapely.geometry import GeometryCollection
from shapely import affinity

from qRiversCode import Width
from qRiversCode import Centerline
from qRiversCode.intersect import intersection
from qRiversCode.GraphSort import getGraph, GraphSort 
from qRiversCode.Picker import PointPicker
from qRiversCode.Curvature import getCurvature


def smoothCenterline(xy, col1, col2, window=3, poly=1):
    smoothed = savgol_filter(
        (xy[col1], xy[col2]), 
        window, 
        poly
    ).transpose()

    return smoothed[:, 0], smoothed[:, 1]


def findEPSG(longitude, latitude):
    zone = math.ceil((longitude + 180) / 6)

    if latitude > 0:
        espg_format = '326{}'
    else:
        espg_format = '327{}'

    if zone < 10:
        zone = '0' + str(zone)

    return espg_format.format(str(zone))


def detrend(df1, df2, d=10):
    x1 = df1['x']
    y1 = df1['y']

    x2 = df2['x']
    y2 = df2['y']

    z = numpy.polyfit(x1, y1, d)
    p = numpy.poly1d(z)

    return y1 - p(x1), y2 - p(x2)


def coordToIndex(xy, xy1, xy2):
    """
    xy are the absolute coordinates of interesections
    xy1 are the cl coordinates at t1
    xy2 are the cl cooridnates at t2

    returns
    i1 intersection indeces at t1
    i2 intersection indees at t2
    """
    # T1 & T2
    i1 = []
    i2 = []
    tree1 = spatial.KDTree(xy1)
    tree2 = spatial.KDTree(xy2)
    for pair in xy:
        # T1
        distance1, n1 = tree1.query(
            pair,
            1
        )
        # T2
        distance2, n2 = tree2.query(
            pair,
            1
        )

        i1.append(n1)
        i2.append(n2)

    return i1, i2


def pickCutoffs(p1, p2, centerline1, centerline2, G):
    """
    Mannually pick cutoff points.
    Algorithm will find the points between the two
    """
    # Pick the points
    fig, ax = plt.subplots(1, 1)
    p1.plot(color='red', ax=ax, label='t1')
    p2.plot(color='blue', ax=ax, label='t2')
    line = ax.scatter(centerline1['coli'], centerline1['rowi'])
    ax.scatter(centerline2['coli'], centerline2['rowi'])
    BC = PointPicker(ax)

    fig.canvas.mpl_connect('pick_event', BC)
    line.set_picker(1)

    axclear = plt.axes([0.81, 0.17, 0.1, 0.055])
    bclear = Button(axclear, 'Clear')
    bclear.on_clicked(BC.clear)

    axnext = plt.axes([0.81, 0.1, 0.1, 0.055])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(BC.next)

    axdone = plt.axes([0.81, 0.03, 0.1, 0.055])
    bdone = Button(axdone, 'Done')
    bdone.on_clicked(BC.done)
    plt.show()

    cutoffdf = pandas.DataFrame()
    cutoffsI = []
    cutoffNums = []
    cutoff_num = 1
    for cutoff in BC.cutoffsI:
    # Find path between the two points
        tree = spatial.KDTree(centerline2[['coli', 'rowi']])
        distance1, neighbor1 = tree.query(cutoff[0])
        distance2, neighbor2 = tree.query(cutoff[1])
        neighbors = [neighbor1, neighbor2]

        cutoffsI.append(neighbors)

        path = numpy.array(
            nx.shortest_path(
                G, 
                source=neighbors[0],
                target=neighbors[1], 
                weight='length'
            )
        )
        points = centerline2.iloc[path]
        points['cutoff'] = cutoff_num
        cutoffdf = cutoffdf.append(points)

        cutoff_num += 1

    cutoffdf = cutoffdf.reset_index(drop=False)

    return cutoffdf, numpy.array(cutoffsI) 


def findCutoffs(I, xy1, xy2, cutoffthresh=3000):
    """
    Finds centerline cutoffs by thresholding the streamwise
    distances between cross-over points.

    Inputs:
    I: nx2 array that has t1 and t2 indices of cross-overs
    xy1: x-y coordinates at t1
    xy2: x-y coordinates at t2
    cutoffthresh: threshold difference in streamwise distance

    returns:
    cutoffs: nx4 array that has cutoff endpoints at t1 and t2
    """
    # iterate through crossover indices
    differences = []
    for prev, cur in zip(I, I[1:]):
        # Find T1 distance
        t1i = [i for i in range(prev[0], cur[0])]
        if (len(t1i) == 1) or (not t1i):
            t1dist = 0
        else:
            t1diff = numpy.diff(xy1[t1i].transpose())
            t1dist = numpy.sum(
                numpy.sqrt(
                    numpy.add(
                        (t1diff[0,:]**2), 
                        (t1diff[1,:]**2)
                    )
                )
            )
        # Find T2 distance
        t2i = [i for i in range(prev[1], cur[1])]
        if (len(t2i) == 1) or (not t2i):
            t2dist = 0
        else:
            t2diff = numpy.diff(xy2[t2i].transpose())
            t2dist = numpy.sum(
                numpy.sqrt(
                    numpy.add(
                        (t2diff[0,:]**2), 
                        (t2diff[1,:]**2)
                    )
                )
            )

        # Find total streamwise distance between intersections
        difference = abs(t1dist - t2dist)
        differences.append(difference)

    # Filter for cutoffs
    jumps = numpy.where(
        numpy.array(differences) > cutoffthresh
    )[0]

    # Save as cutoff object
    cutoffs = numpy.empty((len(jumps), 4))
    for idx, jump in enumerate(jumps):
        cutoffs[idx, 0] = int(I[jump, 0])
        cutoffs[idx, 1] = int(I[jump+1, 0])
        cutoffs[idx, 2] = int(I[jump, 1])
        cutoffs[idx, 3] = int(I[jump+1, 1])

    return cutoffs


def splitIntoChunks(cutoffs, xy1, xy2):
    """
    Splits centerlines into chunks that do not include channel cutoffs

    Inputs:
    cutoffs: indexes of cutoffs
    xy1: x-y coordinates at t1
    xy2: x-y coordinates at t2

    returns:
    segments: non-cutoff channel segments 
    """
    # Split centerlines into chunks that don't include cutoffs
    curindex1 = 0
    curindex2 = 0
    segments1 = []
    segments2 = []
    for idx, cutoff in enumerate(cutoffs):
        segments1.append(xy1[curindex1:int(cutoff[0])])
        segments2.append(xy2[curindex2:int(cutoff[2])])

        curindex1 = int(cutoff[1])
        curindex2 = int(cutoff[3])

        if idx != len(cutoffs)-1:
            nextindex1 = int(cutoffs[idx+1][0])
            nextindex2 = int(cutoffs[idx+1][2])

            segments1.append(xy1[curindex1:nextindex1])
            segments2.append(xy2[curindex2:nextindex2])

            curindex1 = nextindex1
            curindex2 = nextindex2

        else:
            segments1.append(xy1[curindex1:])
            segments2.append(xy2[curindex2:])

    return segments1, segments2


def findMigratedArea(xy1seg, xy2seg):
    """
    Finds the migrated area between two centerlines
    Inputs:

    xy1segs: list of t1 centerline segments (not cutoffs)
    xy2segs: list of t2 centerline segments

    Outputs:

    polygons: list of segment polygons of migrated area
    areas: list of polygon areas
    """
    # Empty list for polygon points
    polygon_points = [] 

    # append all xy points for curve 1
    for xyvalue in xy1seg:
        polygon_points.append([xyvalue[0], xyvalue[1]]) 

    # append all xy points for curve 2 in the reverse order
    for xyvalue in xy2seg[::-1]:
        polygon_points.append([xyvalue[0], xyvalue[1]]) 

    # append the first point in curve 1 again, to it "closes" the polygon
    for xyvalue in xy1seg[0:1]:
        polygon_points.append([xyvalue[0], xyvalue[1]]) 

    return Polygon(polygon_points)


def getDirection(xy, n):
    """
    Calculates UNIT directions for each river coordinate
    This creates two columns:
        - one for the vector direction in LON
        - one for vector direction in LAT
    This is simple case that uses a forward difference model

    Inputs -
    xy (numpy array): Size is n x 2 with the centerline coordinates
    n (int): smoothing to use
    """

    cross_dlon = []
    cross_dlat = []
    tree = spatial.KDTree(xy)
    for idx, row in enumerate(xy):
        distance, neighbors = tree.query(
            [(row[0], row[1])],
            n
        )
        max_distance = numpy.argmax(distance[0])
        max_neighbor = neighbors[0][max_distance]
        min_distance = numpy.argmin(distance[0])
        min_neighbor = neighbors[0][min_distance]

        # Calculate lat and lon distances between coordinates
        distance = [
            (
                xy[max_neighbor][0]
                - xy[min_neighbor][0]
            ),
            (
                xy[max_neighbor][1]
                - xy[min_neighbor][1]
            )
        ]

        # Converts distance to unit distance
        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
        dlon_t, dlat_t = distance[0] / norm, distance[1] / norm
        cross_dlat.append(-1 * dlon_t)
        cross_dlon.append(dlat_t)


    return numpy.vstack([cross_dlon, cross_dlat]).transpose() 


def createCrossSection(location, direction, xprop, yprop):
    xoffset = direction[0] * xprop
    yoffset = direction[1] * yprop

    A = [(location[0] + xoffset), (location[1] + yoffset)]
    B = [(location[0] - xoffset), (location[1] - yoffset)]

    return LineString([A, B])


def getMigrationWindow(location, direction, crosslen, crosswidth):
    # Create box to find migrated area within
    cross_offset = direction * crosslen

    down_direction = numpy.array([
        direction[1],
        -1* direction[0]
    ])
    down_offset = down_direction * crosswidth 

    window_points = numpy.empty([4, 2])
    # 1 corner
    window_points[0, 0] = (
        location[0] + down_offset[0] - cross_offset[0]
    )
    window_points[0, 1] = (
        location[1] + down_offset[1] - cross_offset[1]
    )

    # 3 corner
    window_points[1, 0] = (
        location[0] - down_offset[0] - cross_offset[0]
    )
    window_points[1, 1] = (
        location[1] - down_offset[1] - cross_offset[1]
    )

    # 4 corner
    window_points[2, 0] = (
        location[0] - down_offset[0] + cross_offset[0]
    )
    window_points[2, 1] = (
        location[1] - down_offset[1] + cross_offset[1]
    )

    # 2 corner
    window_points[3, 0] = (
        location[0] + down_offset[0] + cross_offset[0]
    )
    window_points[3, 1] = (
        location[1] + down_offset[1] + cross_offset[1]
    )

    return Polygon(window_points)


def channelMigrationPoly(polyt1, polyt2, centerlinet1, centerlinet2, crosslen, smoothing=5, crosswidth=5):
    # Find the polygon from the difference between the two channels
    diff_poly = polyt2.difference(polyt1)

    # Iterate over centerline time 2
    xy = numpy.array(centerlinet2[['coli', 'rowi']])
    cross_dirs = getDirection(
        xy, 
        smoothing
    )

    migration_distances = []
    for idx, (location, direction) in  enumerate(zip(xy, cross_dirs)):
        window_poly = getMigrationWindow(
            location, 
            direction, 
            crosslen, 
            crosswidth
        )

        migrated_poly = diff_poly.intersection(window_poly)
        migration_distances.append(migrated_poly.area / (2 * crosswidth))

    return migration_distances


def channelMigrationCenterline(line1, line2, crosslen, 
                               crosswidth=5, xcol='rowi', ycol='coli'):

    # Load Data
    x1 = line1[xcol].values
    y1 = line1[ycol].values
    x2 = line2[xcol].values
    y2 = line2[ycol].values
    xy1 = numpy.vstack([x1, y1]).transpose()
    xy2 = numpy.vstack([x2, y2]).transpose()

    # Get Polygon between centerlines
    polygon = findMigratedArea(xy1, xy2).buffer(0)

    cross_dirs = getDirection(xy2, 5)
    xprop = crosslen
    yprop = crosslen

    migration_distances = []
    for idx, (location, direction) in  enumerate(zip(xy2, cross_dirs)):
        window_poly = getMigrationWindow(
            location, 
            direction, 
            crosslen, 
            crosswidth
        )
        migrated_poly = polygon.intersection(window_poly)
        migration_distances.append(migrated_poly.area / (2 * crosswidth))

    return migration_distances



def channelMigrationOld(root, year1, year2, river, cutoffthresh,
                     smoothing, crosslen,
                     xcolumn, ycolumn, inEPSG=4326):
    """
    Original method that uses centerlines to calculate the migrated distances.
    The algorithm will load the centerline data, smooth the centerline,
    find the channel cutoffs, then calculate the migration distances along the
    points of time 1 centerline.

    Inputs:
    root (str) : root path to the working directory
    year1 (str): year of time 1
    year1 (str): year of time 2
    river (str): river to calculate the distances
    curoffthresh (int): maximum distance for non-cutoff migration
    smoothing (int): number of neighbors to smooth the centerline by
    crosslon (int): length of the cross section for measuring the migration
    xcolumn (str): name of the x column (could be lat or lon, or x or y)
    ycolumn (str): name of the y column

    returns:
    df (pandas.DataFrame): dataframe with all positional, migration and cutoff
    information
    """

    year1name = f'{year1}/{river}_{year1}_data.csv'
    year2name = f'{year2}/{river}_{year2}_data.csv'

    year1path = os.path.join(root, year1name)
    year2path = os.path.join(root, year2name)

    year1_df = pandas.read_csv(year1path)
    year2_df = pandas.read_csv(year2path)

    # Smooth Centerline
    year1_df['lon_smooth'], year1_df['lat_smooth'] = smoothCenterline(
        year1_df[['longitude', 'latitude']]
    )
    year2_df['lon_smooth'], year2_df['lat_smooth'] = smoothCenterline(
        year2_df[['longitude', 'latitude']]
    )

    # Reproject to UTM
    # Year 1
    
    print(int(findEPSG(
                year1_df['longitude'].iloc[0],
                year1_df['latitude'].iloc[0]
    )))
    eastings, northings = [], []
    for idx, row in year1_df.iterrows():
        easting, northing = transform_coordinates(
            row['lon_smooth'],
            row['lat_smooth'],
            inEPSG,
            int(findEPSG(
                year1_df['longitude'].iloc[0],
                year1_df['latitude'].iloc[0]
            ))
        )

        eastings.append(easting)
        northings.append(northing)

    year1_df['easting'] = eastings
    year1_df['northing'] = northings

    # Year 2
    eastings, northings = [], []
    for idx, row in year2_df.iterrows():
        easting, northing = transform_coordinates(
            row['lon_smooth'],
            row['lat_smooth'],
            inEPSG,
            int(findEPSG(
                year2_df['longitude'].iloc[0],
                year2_df['latitude'].iloc[0]
            ))
        )

        eastings.append(easting)
        northings.append(northing)

    year2_df['easting'] = eastings
    year2_df['northing'] = northings

    # Detrend
    # Set which to coordinate to be x and y
    # t1
    year1_df['x'] = year1_df[xcolumn]
    year1_df['y'] = year1_df[ycolumn]
    # t2
    year2_df['x'] = year2_df[xcolumn]
    year2_df['y'] = year2_df[ycolumn]

    year1_df['ydetrend'], year2_df['ydetrend'] = detrend(year1_df, year2_df)

    # Save xy positions
    x1 = year1_df['x'].values
    y1 = year1_df['ydetrend'].values
    x2 = year2_df['x'].values
    y2 = year2_df['ydetrend'].values

    # Get coordinates of centerline intersection
    # indexes
    ix, iy = intersection(x1, y1, x2, y2)

    # Turn coordinates into index
    ixy = numpy.vstack([ix, iy]).transpose()
    xy1 = numpy.vstack([x1, y1]).transpose()
    xy2 = numpy.vstack([x2, y2]).transpose()
    oxy = numpy.vstack([x1, year1_df['y']]).transpose()

    # Indexes of intersections at t1 and t2
    i1, i2 = coordToIndex(ixy, xy1, xy2)
    I = numpy.vstack([i1, i2]).transpose()

    # Find cutoffs
    cutoffs = findCutoffs(I, xy1, xy2, cutoffthresh=cutoffthresh)

    # Quick fix
#    cutoffs = cutoffs[2:]
    print(cutoffs)

    if len(cutoffs)> 0:
        # Split centerlines into chunks that don't include cutoffs
        xy1segs, xy2segs = splitIntoChunks(cutoffs, xy1, xy2)
    else:
        xy1segs = [xy1]
        xy2segs = [xy2]

    # Get area between two curves
    polygons = []
    areas = []
    for xy1seg, xy2seg in zip(xy1segs, xy2segs):
        polygon = findMigratedArea(xy1seg, xy2seg)
        area = polygon.area

        polygons.append(polygon)
        areas.append(area)

    # Get migration rate for each point along the curve
    # Get orthogonal direction
    migrations_seg = []
    sections_seg = []
    for idx, (xy1seg, xy2seg) in enumerate(zip(xy1segs, xy2segs)):
        if len(xy1seg) == 0:
            continue

        cross_dirs = getDirection(xy1seg, smoothing)

    #    xprop = numpy.max(xy1seg[:,0]) - numpy.min(xy1seg[:,0])
    #    yprop = numpy.max(xy1seg[:,1]) - numpy.min(xy1seg[:,1])
        xprop = crosslen
        yprop = crosslen

        migration = numpy.zeros((len(xy1seg), 3))
        sections = []
        for jdx, (location, direction) in  enumerate(zip(xy1seg, cross_dirs)):
            # Create the cross-section shape
            section = createCrossSection(
                location, 
                direction, 
                xprop, 
                yprop
            )

            polygon_str = LineString(polygons[idx].exterior)
            li = section.intersection(polygon_str)

            if isinstance(li, MultiPoint):
                xpoints = [l.xy[0] for l in li]
                ypoints = [l.xy[1] for l in li]

                # Handle if cross section intersects more than one bank
                if len(xpoints) > 2:
                    xpoints = xpoints[:1]
                    ypoints = ypoints[:1]

                xdist = numpy.max(xpoints) - numpy.min(xpoints)
                ydist = numpy.max(ypoints) - numpy.min(ypoints)

                sections.append([xpoints, ypoints])

                migration[jdx, 0] = xdist
                migration[jdx, 1] = ydist
                migration[jdx, 2] = math.sqrt((xdist**2) + (ydist**2))

            else:
                migration[jdx, 0] = 0
                migration[jdx, 1] = 0
                migration[jdx, 2] = 0

        migrations_seg.append(migration)
        sections_seg.append(sections)

    # Combine coordinates, cutoffs, and migrated lengths back together 
    xy1full = numpy.zeros((len(xy1), 7))
    startidx = 0

    # Save coordinates
    xy1full[:, 0] = xy1[:,0]
    xy1full[:, 1] = xy1[:,1]
    xy1full[:, 2] = oxy[:,1]

    # Attach migrated distances from segments
    for idx, migration in enumerate(migrations_seg):

        # Only do this more complicated logic if there is a cutoff
        if len(cutoffs) > 0:
            # If it's the first
            if idx == 0:
                # Get indices of the cutoff
                end_cidx = int(cutoffs[idx][0])
                xy1full[:end_cidx, 3] = migration[:, 0]
                xy1full[:end_cidx, 4] = migration[:, 1]
                xy1full[:end_cidx, 5] = migration[:, 2]
                xy1full[:end_cidx, 6] = 0 

            # If it's the last
            elif idx == len(cutoffs):
                start_cidx = int(cutoffs[idx-1][1])
                xy1full[start_cidx:, 3] = migration[:, 0] 
                xy1full[start_cidx:, 4] = migration[:, 1] 
                xy1full[start_cidx:, 5] = migration[:, 2] 
                xy1full[start_cidx:, 6] = 0 

            else:
                start_cidx = int(cutoffs[idx-1][1])
                end_cidx = int(cutoffs[idx][0])
                xy1full[start_cidx:end_cidx, 3] = migration[:, 0] 
                xy1full[start_cidx:end_cidx, 4] = migration[:, 1] 
                xy1full[start_cidx:end_cidx, 5] = migration[:, 2] 
                xy1full[start_cidx:end_cidx, 6] = 0
        else:
            xy1full[:,3] = migration[:,0]
            xy1full[:,4] = migration[:,1]
            xy1full[:,5] = migration[:,2]
            xy1full[:,6] = 0 

    # Save the cutoffs
    for cutoff in cutoffs:
        xy1full[int(cutoff[0]):int(cutoff[1]), 6] = 1

    df = pandas.DataFrame(
        xy1full, 
        columns=[
            'x', 
            'ydetrend', 
            'y', 
            'Xmigration',
            'Ymigration',
            'MagMigration',
            'cutoff', 
        ]
    )
    df['cutoff'] = df['cutoff'].astype('bool')

    return df


def channelMigrationPolyOld(polyt1, polyt2, centerlinet1, centerlinet2,
                         crosslen=10, smoothing=5):
    # Find the polygon from the difference between the two channels
    diff_poly = polyt2.difference(polyt1)

    # Iterate over centerline time 2
    xy = numpy.array(centerlinet2[['coli', 'rowi']])
    cross_dirs = getDirection(
        xy, 
        smoothing
    )

    migration_distances = []
    p = gpd.GeoSeries(diff_poly)
    for idx, (location, direction) in  enumerate(zip(xy, cross_dirs)):
        # Create the cross-section shape
        section = createCrossSection(
            location, 
            direction, 
            crosslen, 
            crosslen 
        )

        # Intersected area between the cross section and the migrated area
        migration = section.intersection(diff_poly.buffer(0))

        # If the cross-section intersects multiple areas of change 
        if isinstance(migration, MultiLineString):
            dist = 0
            for string in migration:
                a = numpy.array([string.xy[0][0], string.xy[1][0]])
                b = numpy.array([string.xy[0][1], string.xy[1][1]])
                dist += numpy.linalg.norm(a-b)
        # If the cross-section intersects one change-area
        elif isinstance(migration, LineString):
            a = numpy.array([migration.xy[0][0], migration.xy[1][0]])
            b = numpy.array([migration.xy[0][1], migration.xy[1][1]])
            dist = numpy.linalg.norm(a-b)
        # If the cross-section does not intersect any change-area
        else:
            dist = None

        migration_distances.append(dist)

    # Save as a new column
    centerlinet2['migration'] = migration_distances

    return centerlinet2


def getCutoffPoints(cutoff_path, graph_path, clpath2):
#    cutoff_path = 'cutoff/Pearl_1984_2020_cutoff.shp'
#    clpath2 = 'width/Pearl_2020_Width.csv'
#    graph_path = 'graph/Pearl_2020_graph'

    # Load graph
    G = nx.read_gpickle(graph_path)

    # Get cutoff end points from shapefile
    ds = ogr.Open(cutoff_path)
    lyr = ds.GetLayer()

    dfn = lyr.GetLayerDefn()
    nfields = dfn.GetFieldCount()
    fields = []
    for i in range(nfields):
        fields.append(dfn.GetFieldDefn(i).GetName())
    fields.append('X')
    fields.append('Y')

    data = {field: [] for field in fields}
    for feat in lyr:
        attributes = feat.items()
        geom = feat.GetGeometryRef()
        for key, attr in attributes.items():
            data[key].append(attr)

        data['X'].append(geom.GetX())
        data['Y'].append(geom.GetY())

    cutoff = pandas.DataFrame(data=data)

    # Get all points within cutoffs
    centerline2 = pandas.read_csv(clpath2)
    tree = spatial.KDTree(centerline2[['x', 'y']])
    num_cutoffs = cutoff['i'].unique()

    cutoffdf = pandas.DataFrame()
    for icut in num_cutoffs:
        icutdf = cutoff[cutoff['i'] == icut]
        usxy = numpy.array(icutdf[icutdf['usds'] == 'us'][['X', 'Y']])
        dsxy = numpy.array(icutdf[icutdf['usds'] == 'ds'][['X', 'Y']])

        us_distance, us_neighbor = tree.query(usxy)
        ds_distance, ds_neighbor = tree.query(dsxy)

        path = numpy.array(
            nx.shortest_path(
                G, 
                source=us_neighbor[0],
                target=ds_neighbor[0], 
                weight='length'
            )
        )
        points = centerline2.iloc[path]
        points['cutoff'] = i 
        cutoffdf = cutoffdf.append(points)

    return cutoffdf.reset_index(drop=False)


def qMigration(time1, polypath1, clpath1, graphpath1,
               time2, polypath2, clpath2, graphpath2,
               cutoff):
    """
    year: int
    polypath: str path
    clpath: str path
    graphpath: str path
    cutoff: numpy ar
    """

    # Load river polygons
    with open(polypath1, 'rb') as f:
        poly1 = pickle.load(f)
    with open(polypath2, 'rb') as f:
        poly2 = pickle.load(f)

    # Load river centerlines
    cl1 = pandas.read_csv(clpath1)
    cl2 = pandas.read_csv(clpath2)

    # Load graphs
    graph1 = nx.read_gpickle(graphpath1)
    graph2 = nx.read_gpickle(graphpath2)

    # Find curvature
    cl2['curvature'] = getCurvature(graph2, cl2)

    # GraphSort Centerlines
    clSort1 = GraphSort(
        graph1,
        cl1,
    )

    clSort2 = GraphSort(
        graph2,
        cl2,
    )

    # Get migration from two sorted centerlines
    cl2['migration_centerline'] = None
    clSort2['migration_centerline'] = channelMigrationCenterline(
        clSort1,
        clSort2,
        cl2['width'].mean()
    )
    
    # Set values in other dataframe
    for idx, row in clSort2.iterrows():
        cl2.at[
            row['idx'],
            'migration_centerline'
        ] = row['migration_centerline']

    # Get migration from differencing two polys
    cl2['migration_poly'] = channelMigrationPoly(
        poly1, 
        poly2, 
        cl1, 
        cl2, 
        cl2['width'].mean()
    )

    # Get time between two images
    dt1 = datetime.strptime(time1, '%Y-%m-%d')
    dt2 = datetime.strptime(time2, '%Y-%m-%d')
    span = round((dt2 - dt1).days / 365, 2)

    cl2['span'] = span 

    # Remove cutoff points
    cl2['cutoff'] = 0
    if len(cutoff):
        for i, row in cutoff.iterrows():
            cl2.at[row['index'], 'cutoff'] = row['cutoff']
            cl2.at[row['index'], 'migration_centerline'] = None
            cl2.at[row['index'], 'migration_poly'] = None

    return cl2


