#!/usr/bin/env python
# coding: utf-8
import os
import glob
from datetime import datetime
from collections import OrderedDict
from itertools import product
from typing import List
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

import satpy

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom

import rasterio
from rasterio.warp import reproject, Resampling, transform
from rasterio.control import GroundControlPoint
from rasterio.transform import xy
from pyresample.geometry import AreaDefinition

import argparse
import shutil
import warnings

warnings.filterwarnings("ignore", module="pyproj")
warnings.filterwarnings("ignore", module="dask")

parser = argparse.ArgumentParser(description='Reproject the native EUMETSAT data.')
parser.add_argument('startyear', type=int,
                    help='first year to process (inclusive)')
parser.add_argument('startmonth', type=int,
                    help='first month to process (inclusive)')
parser.add_argument('endyear', type=int,
                    help='last year to process (inclusive)')
parser.add_argument('endmonth', type=int,
                    help='last month to process (inclusive)')
parser.add_argument('--startday', dest='startday', type=int, default=1,
                    help='first day to process (inclusive)')
parser.add_argument('--endday', dest='endday', type=int, default=-1,
                    help='last day to process (inclusive)')
args = parser.parse_args()

# these are all the channels available
all_channels = ['HRV', 'IR_016', 'IR_039','IR_087', 'IR_097', 'IR_108', 'IR_120', 
                'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

# reproject these specific channels
reproject_channels = all_channels

STAGING_PATH = os.path.expanduser('~/staging')
TEMP_PATH = os.path.join(STAGING_PATH, 'tmp')
os.makedirs(STAGING_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

GS_BUCKET = 'solar-pv-nowcasting-data'
NATIVE_GS_location = "satellite/EUMETSAT/SEVIRI_RSS/native"
REPROJECTED_GS_location = "satellite/EUMETSAT/SEVIRI_RSS/OSGB36"


SRC_CRS = {
        'proj': 'geos',  # Geostationary
        'lon_0': 9.5,
        'a': 6378169.0,
        'b': 6356583.8,
        # The EUMETSAT docs say "The distance between spacecraft and centre of earth is 42,164 km. The idealized earth
        # is a perfect ellipsoid with an equator radius of 6378.1690 km and a polar radius of 6356.5838 km." 
        # The projection used by SatPy expresses height as height above the Earth's surface (not distance
        # to the centre of the Earth).
        'h': 35785831.00,  # Height of satellite above the Earth's surface
        'units': 'm'  # meters
}


# Hard-code destination transform 
# Taken as the same grid as Metoffice NWP
#    > see http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
DST_CRS = 'EPSG:27700'
METERS_PER_PIXEL = 2000
WEST=-239_000
SOUTH=-185_000
EAST=857_000
NORTH=1223_000 
DST_WIDTH = int(abs((EAST - WEST)) / METERS_PER_PIXEL)
DST_HEIGHT = int(abs((NORTH - SOUTH)) / METERS_PER_PIXEL)


DST_TRANSFORM = rasterio.transform.from_bounds(
    west=WEST, south=SOUTH, east=EAST, north=NORTH, width=DST_WIDTH, height=DST_HEIGHT)

def rmtree(path, remove_empty_parents=True):
    """
    remove tree and empty parent directories of path
    """
    shutil.rmtree(path)
    if remove_empty_parents:
        path_chunks = path.split('/')
        path_chunks[0] = '/'+path_chunks[0]
        while path_chunks:
            path_chunks.pop(-1)
            try:
                os.rmdir(os.path.join(*path_chunks))
            except OSError:
                break
    return
    
def get_directory_for_datetime(sat_imagery_path: str, dt: datetime) -> str:
    """
    Args:
        sat_imagery_path:
        dt: the datetime for the requested image.  Will return
            the directory which is within 5 minutes of the requested image.
    Returns:
        Directory string
        
    Raises:
        FileNotFoundError
    """
    hour_path = dt.strftime("%Y/%m/%d/%H")
    hour_path = os.path.join(sat_imagery_path, hour_path)
    print('looking in ', hour_path)
    
    # Get a list of subdirectories containing minutes
    try:
        _, minute_dirs, _ = next(os.walk(hour_path))
    except StopIteration:
        raise FileNotFoundError
    minutes_with_images = np.array(minute_dirs, dtype=int)
    
    # Quantize dt.minute to 5-minute intervals
    minute_lower_bound = (dt.minute // 5) * 5
    minute_upper_bound = minute_lower_bound + 5
    
    # Find matching directory for the minutes
    selection_condition = (
        (minute_lower_bound <= minutes_with_images) & 
        (minutes_with_images < minute_upper_bound))
    idx = np.flatnonzero(selection_condition)
    
    # Sanity check
    if idx.size == 0:
        raise FileNotFoundError(2, 'No minute directory for datetime {} under "{}"'.format(dt, hour_path))
    elif idx.size > 1:
        raise RuntimeError(
            'Found > 1 directories with images for datetime {}.'
            '  Base dir = "{}".  Subdirs found = {}'
            .format(dt, hour_path, minute_dirs))
        
    selected_minute_dir = minutes_with_images[idx[0]]
    selected_minute_dir = '{:02d}'.format(selected_minute_dir)
    selected_minute_dir = os.path.join(hour_path, selected_minute_dir)
    
    return selected_minute_dir


def get_image_filename_for_datetime(
    sat_imagery_path: str, 
    dt: datetime,
    pattern: str = 'MSG*.nat.bz2',
    ) -> str:
    
    image_path = get_directory_for_datetime(sat_imagery_path, dt)
    files = glob.glob(os.path.join(image_path, pattern))[:1]
    error_str = 'file matching "{}" in "{}" for datetime {}.'.format(pattern, image_path, dt)
    if len(files) == 0:
        raise FileNotFoundError(2, 'No ' + error_str)
    if len(files) > 1:
        raise RuntimeError('Found > 1 ' + error_str + '  Expected only one match.', files)
    image_filename = files[0]
    return image_filename


def decompress(full_bzip_filename: str) -> str:
    base_bzip_filename = os.path.basename(full_bzip_filename)
    base_nat_filename = os.path.splitext(base_bzip_filename)[0]
    full_nat_filename = os.path.join(TEMP_PATH, base_nat_filename)
    if os.path.exists(full_nat_filename):
        os.remove(full_nat_filename)
    with open(full_nat_filename, 'wb') as nat_file_handler:
        process = subprocess.run(
            ['pbzip2', '--decompress', '--keep', '--stdout', full_bzip_filename],
            stdout=nat_file_handler)
    process.check_returncode()
    return full_nat_filename


def combine_attributes(hrv_attrs, visir_attrs):
    """Very rough function to port some of the attributes.
    Some need to be modified so can be saved to netcdf."""
    overrided = {'resolution'}
    drop = set() #{'area', 'ancillary_variables', }
    if hrv_attrs is not None and visir_attrs is not None:
        attrs = {k:hrv_attrs[k] for k in set(hrv_attrs.keys())&set(visir_attrs.keys())-overrided-drop
                 if hrv_attrs[k]==visir_attrs[k]}
        attrs['HRV_original_source'] = {k:hrv_attrs[k] for k in set(hrv_attrs.keys())-set(attrs.keys())-drop}
        attrs['VIS_IR_original_source'] = {k:visir_attrs[k] for k in set(visir_attrs.keys())-set(attrs.keys())-drop}    
    elif hrv_attrs is not None:
        attrs = {k:hrv_attrs[k] for k in set(hrv_attrs.keys())-overrided-drop}
        attrs['HRV_original_source'] = {k:hrv_attrs[k] for k in set(hrv_attrs.keys())-set(attrs.keys())-drop}
    else:
        attrs = {k:visir_attrs[k] for k in visir_attrs.keys() if k not in overrided}
        attrs['HRV_original_source'] = {k:visir_attrs[k] for k in set(visir_attrs.keys())-set(attrs.keys())-drop}
    attrs['projection'] = DST_CRS
    if 'start_time' in attrs.keys():
        attrs['start_time'] = attrs['start_time'] .isoformat()
    if 'end_time' in attrs.keys():
        attrs['end_time'] = attrs['end_time'] .isoformat()
    attrs = {k:str(v) for k, v in attrs.items()}
    return attrs


def reproject_xr(raw_ds: xr.Dataset, area_extent: List[float], dt: datetime) -> xr.Dataset:

    raw_image = raw_ds.to_array().values
    raw_channels, raw_height, raw_width = raw_image.shape
    channel_names = np.array([k for k in raw_ds.keys()], dtype=str)
    
    # Make array of NaNs to accept transformed image
    dst_shape = (raw_channels, DST_HEIGHT, DST_WIDTH)
    dst_array = np.full(dst_shape, np.nan, dtype=np.float32)
    
    # create ground control points from extent
    left, bottom, right, top = area_extent
    ground_control_points = [
        GroundControlPoint(row=0, col=0, x=left, y=top, id='top_left'),
        GroundControlPoint(row=raw_height, col=0, x=left, y=bottom, id='bottom_left'),
        GroundControlPoint(row=raw_height, col=raw_width, x=right, y=bottom, id='bottom_right'),
        GroundControlPoint(row=0, col=raw_width, x=right, y=top, id='top_right'),
    ]
    
    # Reproject
    reproject(
        raw_image,
        dst_array,
        src_crs=SRC_CRS,
        dst_crs=DST_CRS,
        gcps=ground_control_points,
        dst_transform=DST_TRANSFORM,
        num_threads=8,
        resampling=Resampling.cubic,
        src_nodata=np.nan)
    
    # Get X's and Y's (coordinates of each column and row, respectively)
    n = max(DST_HEIGHT, DST_WIDTH)
    xs, ys = xy(
        transform=DST_TRANSFORM,
        rows=np.arange(n),
        cols=np.arange(n))
    ys = ys[:DST_HEIGHT]
    xs = xs[:DST_WIDTH]
    
    # The Xs and Ys are integers represented as floats even though they
    # are integer values, so convert to ints.
    ys = np.int64(ys)
    xs = np.int64(xs)
    

    dims = OrderedDict()
    dims['time'] = [dt]
    dims['y'] = ys#+5000
    dims['x'] = xs#-3000

    ds = xr.Dataset(
        {c:(dims.keys(), x[np.newaxis,...]) for x, c in zip(dst_array, channel_names)},
        coords=dims,
        attrs=raw_ds.attrs)
    
    return ds

    
def get_reprojected_image(sat_imagery_path, dt, columns=['HRV',], apply_hand_tunig=False) -> xr.DataArray:
    # check columns valid
    assert set(columns)-set(all_channels)==set(), 'columns chosen not valid'
    
    # get filename and unzip
    full_bzip_filename = get_image_filename_for_datetime(sat_imagery_path, dt)
    full_nat_filename = decompress(full_bzip_filename)
    
    # HRV has different grid so must be treated seperatrely
    include_hrv = 'HRV' in columns
    columns_visir = [c for c in columns if c!='HRV']
    include_visir = bool(len(columns_visir))
    
    # load all required datasets
    scene = satpy.Scene(
        filenames=[full_nat_filename],
        reader='seviri_l1b_native')
    scene.load(columns)
    
    # handler is instance of https://satpy.readthedocs.io/en/latest/_modules/satpy/readers/seviri_l1b_native.html
    handler = scene.readers['seviri_l1b_native'].file_handlers['native_msg'][0]
    # alternatively can make a new handler as below
    #handler = satpy.readers.seviri_l1b_native.NativeMSGFileHandler(full_nat_filename, {}, None)
    
    # need to manually set this to true so .get_area_extent() returns correct values
    handler.mda['is_full_disk']=True
    
    # function to select dataset_id given it's name
    select_dataset_id_by_name = lambda name : list(filter(lambda x: x.name==name, scene.all_dataset_ids()))
    
    # HRV has upper and lower area extent. lower seems to be correct and matches with Jack's hardcoded numbers
    _, area_extent_hrv, *_ = handler.get_area_extent(select_dataset_id_by_name('HRV')[1])
    if apply_hand_tunig:
        # these values were tuned by eye by plotting reprojected map
        left, bottom, right, top = area_extent_hrv
        xoff = -500
        yoff = 2000
        area_extent_hrv = left+xoff, bottom+yoff, right+xoff, top+yoff
    
    # Confirm that all visir extents are the same
    if include_visir:
        area_extents_visir = [handler.get_area_extent(select_dataset_id_by_name(column)[0]) for column in columns_visir]
        assert all([area_extent==area_extents_visir[0] for area_extent in area_extents_visir]), 'VIS_IR area extents not the same'
        area_extent_visir = area_extents_visir[0]
        if apply_hand_tunig:
            left, bottom, right, top = area_extent_visir
            # these values were tuned by eye by plotting reprojected map
            xoff = 0
            yoff = -1000
            area_extent_visir = left+xoff, bottom+yoff, right+xoff, top+yoff
    
    # HRV is on different grid to other data. Separate it out
    ds_hrv = scene.to_xarray_dataset(datasets=['HRV']) if include_hrv else None
    ds_visir = scene.to_xarray_dataset(datasets=columns_visir) if include_visir else None
    
    # clean up unpacked data
    os.remove(full_nat_filename)
    
    # reproject
    vis_ir_columns = [c for c in columns if c!='HRV']
    if include_hrv:
        ds_reprojected_hrv = reproject_xr(ds_hrv, area_extent_hrv, dt)
    if include_visir:
        ds_reprojected_visir = reproject_xr(ds_visir, area_extent_visir, dt)
    
    if include_hrv and include_visir:
        # combine
        ds = ds_reprojected_hrv.merge(ds_reprojected_visir)
    elif include_hrv:
        ds = ds_reprojected_hrv
    else:
        ds = ds_reprojected_visir
    
    ds.attrs = combine_attributes(ds_hrv.attrs if ds_hrv is not None else None, 
                                  ds_visir.attrs if ds_visir is not None else None, )
    return ds

one_min = pd.Timedelta(minutes=1)

# copy the files from cloud storage
for year in range(args.startyear, args.endyear+1):
    startmonth = args.startmonth if year==args.startyear else 1
    endmonth = args.endmonth if year==args.endyear else 12
    for month in range(startmonth, endmonth+1):
        
        # download this batch of data
        out = subprocess.Popen(["gsutil", "ls", f"gs://{GS_BUCKET}/{NATIVE_GS_location}/{year:04}/{month:02}/"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        days = sorted([int(f[-3:-1]) for f in str(stdout).split('\\n')[:-1]])
        print(days)
        
        if year==args.startyear and month==args.startmonth:
            days = [d for d in days if d>=args.startday]
        if year==args.endyear and month==args.endmonth and args.endday!=-1:
            days = [d for d in days if d<=args.endday]
        
        print('days', days)

        ds_list = []
        # loop over month data and do reprojections
        for day in days:
            
            # download a day's worth of data
            downl_path = f"{STAGING_PATH}/download/{year:04}/{month:02}/{day:02}"
            upl_path = f"{STAGING_PATH}/upload/{year:04}/{month:02}/{day:02}"
            os.makedirs(downl_path, exist_ok=True)
            os.makedirs(upl_path, exist_ok=True)
            
            download_command = f"gsutil -m cp -r gs://{GS_BUCKET}/{NATIVE_GS_location}/{year:04}/{month:02}/{day:02}/* {downl_path}/."
            os.system(download_command)
            
            # project the day's data
            for hour in sorted([int(f[-2:]) for f in glob.glob(f"{downl_path}/*")]):
                
                for minute in sorted([int(f[-2:]) for f in glob.glob(f"{downl_path}/{hour:02}/*")]):
                    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
                    print(dt)
                    
                    try:
                        ds = get_reprojected_image(os.path.join(STAGING_PATH, 'download'), 
                                                       dt, columns=reproject_channels, 
                                                       apply_hand_tunig=True)
                    except subprocess.CalledProcessError as e:
                        print('`subprocess` error.\n', e, '\nskipping this file.\n')
                        continue
                    
                    if ds_list and (
                        (ds.time+one_min).dt.hour.values[0]!=
                        (ds_list[-1].time+one_min).dt.hour.values[0]
                        ):
                        print('saving...')
                        ds_hour = xr.concat(ds_list, dim='time')
                        ds_hour.to_netcdf(f"{upl_path}/{year:04}-{month:02}-{day:02}T{hour:02}_allchannels.nc",)
                        ds_list = [ds]
                    else:
                        ds_list.append(ds)
        
        
            # upload and clear up
            rmtree(downl_path)
            # make folder to hold data 
            upload_command = f"gsutil -m cp -r {upl_path} gs://{GS_BUCKET}/{REPROJECTED_GS_location}/{year:04}/{month:02}/{day:02}"
            print(upload_command)
            os.system(upload_command)
            rmtree(upl_path)
        
        
        
        




