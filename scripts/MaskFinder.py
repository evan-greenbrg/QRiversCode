import argparse
import getopt
import sys

import ee
import ee.mapclient
import folium
from folium import plugins
import geopandas
import numpy as np
import pandas

from RivWidthCloud import rwc_landsat
from RivWidthCloud import functions_landsat
from RivWidthCloud import functions_river
from RivWidthCloud import functions_centerline_width


# ee.Authenticate()
ee.Initialize()
# folium.Map.add_ee_layer = add_ee_layer

# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self) 
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        else:
            folium.GeoJson(
                data = v,
                name = k
            ).add_to(self)
    except:
        print("Could not display {}".format(name))


# Map Admin stuff
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )
}
folium.Map.add_ee_layer = add_ee_layer

# Start Main Body - Main algo
def maskL8sr(image):
    # Bits 3 and 5 are cloud shadow and cloud
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get pixel QA band
    qa = image.select('BQA')
    # Both flags should be zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0)
    )

    return image.updateMask(mask)



def getImage(collection, polygon, begin, end, cloud_thresh=50):

    # Filter image collection by
    return collection.map(
        maskL8sr
    ).filterDate(
        begin, end 
    ).filterMetadata(
        'CLOUD_COVER_LAND', 'less_than', cloud_thresh
    ).median().clip(
        polygon
    )


def getPolygon(i1, i2, idxs):
    nw = idxs[(idxs['idx']==i2) & (idxs['ew']=='w')]
    sw = idxs[(idxs['idx']==i1) & (idxs['ew']=='w')]
    se = idxs[(idxs['idx']==i1) & (idxs['ew']=='e')]
    ne = idxs[(idxs['idx']==i2) & (idxs['ew']=='e')]

    coords = [
        float(nw['longitude']), float(nw['latitude']),
        float(sw['longitude']), float(sw['latitude']),
        float(se['longitude']), float(se['latitude']),
        float(ne['longitude']), float(ne['latitude']),
    ]

    return ee.Geometry.Polygon(coords)


def startWidthTask(image, args, exportPrefix):
    rwc = rwc_landsat.rwGenSR(
        scale=args['scale'],
        WATER_METHOD=args['water_method'],
        MAXDISTANCE=args['maxdistance'],
        FILL_SIZE=args['fill_size'],
        MAXDISTANCE_BRANCH_REMOVAL=args['maxdistance_branch_removal']
    )

    widthOut = rwc(image)

    taskWidth = (
        ee.batch.Export.table.toDrive(
            collection = widthOut,
            description = exportPrefix,
            folder = args['output_folder'],
            fileNamePrefix = exportPrefix,
            fileFormat = args['format']
        )
    )

    taskWidth.start()


def startMaskTask(image, args, exportPrefix, polygon, desc, bucket='earth-engine-rivmap'):
    mask = rwc_landsat.getMask(
        scale=30,
        water_method=args['water_method'],
        maxdistance=args['maxdistance'],
        fill_size=args['fill_size'],
        maxdistance_branch_removal=args['maxdistance_branch_removal']
    )

    maskOut = mask(image).select('riverMask')

    taskMask = ee.batch.Export.image.toCloudStorage(
        image=maskOut,             
        region=polygon,
        maxPixels=1e13,
        scale=30,
        crs='EPSG:4326',
        description=desc,
        bucket=bucket,
        fileNamePrefix=exportPrefix,
        fileFormat='GeoTIFF',
        formatOptions={
            'cloudOptimized': True
        }
    )

    taskMask.start()

    return maskOut

def startImageTask(image, args, exportPrefix, polygon, desc, bucket='earth-engine-rivmap'):
    taskImage = ee.batch.Export.image.toCloudStorage(
        image=image.select(
            ['uBlue', 'Blue', 'Green', 'Red', 'Swir1', 'Nir', 'Swir2']
        ),
        region=polygon,
        maxPixels=1e13,
        scale=30,
        crs='EPSG:4326',
        description=desc,
        bucket=bucket,
        fileNamePrefix=exportPrefix,
        fileFormat='GeoTIFF',
        formatOptions={
            'cloudOptimized': True
        }
    )

    taskImage.start()


# Load image collection of all landsat images
allLandsat = ee.ImageCollection(
    functions_landsat.merge_collections_std_bandnames_collection1tier1_sr()
)
lanVis = {
    'bands': ['Red', 'Green', 'Blue'],
    'min': 0,
    'max': 3000,
#    'palette':['225ea8','41b6c4','a1dab4','ffffcc']
}

# Get args for images
args = {
    'format':'csv',
    'water_method': 'Zou2018',
    'maxdistance': 4000,
    'scale': 120,
    'fill_size': 333,
    'maxdistance_branch_removal': 500,
    'output_folder': 'GEE',
}

# Get all the indexes to iterate over
idxs = pandas.read_csv('MissLeclairImageIndxs.txt')
iss = idxs['idx'].unique()
iss = iss[iss != max(iss)]

# Get all years to iterate over
years = range(1984, 2022, 2)
years = [2018]
river = 'MissLeclair'

# Pull the image masks
for year in years:
    print('year: ', year)
    for i in iss:
        print('idx: ', i)
        i1 = i
        i2 = i + 1
        polygon = getPolygon(i1, i2, idxs)

        begin = f'{year}-01-01'
        end = f'{year}-12-30'
        image = getImage(allLandsat, polygon, begin, end)

        name = f'{river}/image/{year}/idx{i1}/{river}_{year}_{i1}_image'
        desc = f'{river}_{year}_{i1}_image'
        ImageOut = startImageTask(image, args, name, polygon, desc)
