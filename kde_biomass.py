#  Regrid Biomass data to 
ds = xr.open_dataset("/burg/glab/users/os2328/data/SMOS_IC_xr.zarr") # /burg/glab/users/os2328/data/VOD_project/VOD-IB/smap_ib_xr_ver2.zarr") # all_vod.nc")
ds = ds.sortby(ds.time)
ds = ds.sel(time=slice('2019-01-01', '2019-12-31'))
ds = ds.resample(time='1Y').mean(dim='time')
ds = ds.transpose('time', "lat", "lon")
#ds = ds[['time', "lat", "lon",  'str_mean']]
ds = ds[['time', "lat", "lon",  'Optical_Thickness_Nad']]

lats = np.unique(ds['lat'])
lons = np.unique(ds['lon'])


ds_out = xr.Dataset(
    {
        "lat": (["lat"], lats),
        "lon": (["lon"], lons),
    }
)
print(ds_out)
regridder = xe.Regridder(biomass, ds_out, "bilinear", periodic = True)
mean_out = regridder(biomass['biomass_saatchi'])
print(mean_out)

ds['biomass_saatchi'] = mean_out
mean_out2 = regridder(biomass['biomass_CCI'])
print(mean_out2)

ds['biomass_cci'] = mean_out2

y2 = sample['biomass_saatchi'].values
xy2 = np.vstack([x, y2])
z2 = gaussian_kde(xy2)(xy2)
idx2 = z2.argsort()
x2 = x[idx2]
y2 = y2[idx2]
z2 = z2[idx2]


y3 = sample['canopy'].values
xy3 = np.vstack([x, y3])
z3 = gaussian_kde(xy3)(xy3)
idx3 = z3.argsort()
x3 = x[idx3]
y3 = y3[idx3]
z3 = z3[idx3]



out_p = pd.DataFrame({'x_cci': x1, 'y_cci':y, 'z_cci':z, 'x_saa': x2, 'y_saa':y2, 'z_saa':z2, 'x_ca': x3, 'y_ca':y3, 'z_ca':z3})
out_p.to_pickle('/burg/glab/users/os2328/data/VOD_project/vod_ic_biomass.pkl', protocol = 4)
quit()
