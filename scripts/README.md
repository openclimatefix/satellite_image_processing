The following scripts were used to take the native SEVIRI satellite data as it 
is stored in our google cloud store in .nat.bz2 format, and reproject to the 
OSGB36 grid format (transverse mercator grid centred on the UK) with 2km grid
grid spacing. This is the exact grid used by the MetOffice UKV modeloutput and
so allows for easy concatenation of these datasets. 

The script `reproject.py` downloads, processes and reprojects the SEVIRI data
and then uploads them in netcdf format to the cloud store. The files it creates
represent one hour of 5min interval data for all channels on the instrument. 
For example the file 2019-02-03T03_allchannels.nc would contain datetimes
`[2019-02-03T02:59, 2019-02-03T03:04, 2019-02-03T03:09, ..., 2019-02-03T03:54]`

The script `sat_netcdf_to_zarr.py` downloads, converts and uploads the processed 
netcdf files generated with the reproject script. This was done mainly as
converting to zarr was decided after the reprojecing was well under way. However
an added bonus is that saving to the netcdf archive can be done in parallel and
so the reproject script can be run in multiple instances. Saving to zarr in such
a way is much more involved of a process and not readily supported yet by xarray.