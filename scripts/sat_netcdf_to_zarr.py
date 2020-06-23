"""Simple script to move the EUMETSAT netcdf files which were converted using
`reproject_gcp.py` to a zarr store.
"""

import os
import xarray as xr
import zarr
import subprocess

################################################################################
# USER DEFINED
################################################################################

GS_BUCKET = 'solar-pv-nowcasting-data'
GS_NETCDF_PATH = "satellite/EUMETSAT/SEVIRI_RSS/OSGB36"
GS_ZARR_PATH = "satellite/EUMETSAT/SEVIRI_RSS/OSGB36/zarr"

LOCAL_STAGING_PATH = os.path.expanduser('~/staging')

local_netcdf_path = os.path.join(LOCAL_STAGING_PATH, 'netcdf_dump')
local_zarr_path = os.path.join(LOCAL_STAGING_PATH, 'zarr_store')

################################################################################
# BASIC SETUP
################################################################################

# make directories if needed
os.makedirs(local_netcdf_path, exist_ok=True)
os.makedirs(local_zarr_path, exist_ok=True)

# Find which days we have netdcdf data for 
out = subprocess.Popen(["gsutil", "ls", "-d", f"gs://{GS_BUCKET}/{GS_NETCDF_PATH}/201[8-9]/[0-1][0-9]/[0-3][0-9]"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.STDOUT)
stdout, stderr = out.communicate()
netcdf_dates = [[int(n) for n in s.split('/')[-4:-1]] for s in stdout.decode().split('\n')[:-1]]

################################################################################
# FUNCTIONS
################################################################################

def download_sat_netcdf(year, month, day):
    """Download a full day's woth of netcdf files ~5GB"""
    filepath = f"{GS_BUCKET}/{GS_NETCDF_PATH}/{year}/{month:02}/{day:02}/*"
    dl_cmd = f"gsutil -m cp gs://{filepath} {local_netcdf_path}/."
    os.system(dl_cmd)
    return


def netcdf_to_zarr(year, month, day, create_new=False):
    """Download netcdfs from GCS and reupload to GCS zarr via staging locally"""
    # download and open new dataset
    download_sat_netcdf(year, month, day)
    ds = xr.open_mfdataset(local_netcdf_path+"/*.nc", combine='by_coords').sortby('time')
    
    # Combine variables into new dimension
    # then chunk single items in time and no chunking in other dimensions.
    ds = ds.to_array()
    ds.name = 'stacked_eumetsat_data'
    ds = ds.to_dataset()
    ds = ds.transpose('time', 'y', 'x', 'variable')
    ds = ds.chunk(dict(variable=-1, time=1, x=-1, y=-1))
    
    # get encodings and save to zarr
    if create_new:
        encoding = { var_name: {
                'filters': [zarr.Delta(dtype='float32')],
                'compressor': zarr.Blosc(cname='zstd', 
                                         clevel=4, 
                                         shuffle=zarr.Blosc.AUTOSHUFFLE)}
                    for var_name in list(ds.variables)}
        ds.to_zarr(local_zarr_path, consolidated=True, encoding=encoding) 
    
    else: # if we are appending to an existing zarr file use this
        ds.to_zarr(local_zarr_path, append_dim='time', consolidated=True)
    
    # upload zarr to GCS
    upl_cmd = f"gsutil -m cp -r {local_zarr_path}/* gs://{GS_BUCKET}/{GS_ZARR_PATH}"
    os.system(upl_cmd)
    upl_cmd = f"gsutil -m cp {local_zarr_path}/.z* gs://{GS_BUCKET}/{GS_ZARR_PATH}"
    os.system(upl_cmd)
    
    # remove local zarr data
    empty_local_zarr(local_zarr_path, list(ds.keys()))
    
    # remove local netcdfs
    netdf_del_cmd = f"rm {local_netcdf_path}/*"
    os.system(netdf_del_cmd)
    

def empty_local_zarr(local_zarr_path, variables):
    '''Empty the Zarr archive of its data (but not its metadata).
    Arguments:
        - name: the name of the archive.
        - variables: the name of the variable(s) to empty
    '''
    for v in variables:
        os.system(f'rm {local_zarr_path}/{v}/[0-9]*')
    
    
create_new = True
# loop through all files and convert and upload as zarr
for year, month, day in netcdf_dates:
    netcdf_to_zarr(year, month, day, create_new=create_new)
    create_new = False
    
    
    