import os
import numpy as np
import xarray as xr

backend_dir = os.path.dirname(__file__)
nc_path = os.path.join(backend_dir, "era5_india_latest.nc")

print("Generating NetCDF ERA5 dataset...")
lats = np.arange(6.0, 38.25, 0.25)
lons = np.arange(68.0, 98.25, 0.25)
times = np.arange(1, 6)

shape = (len(times), len(lats), len(lons))
rng = np.random.default_rng(42)

lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
base_temp = 42.0 - (lat_mesh - 8.0) * 0.7

temp_data = np.zeros(shape, dtype=np.float32)
pres_data = np.zeros(shape, dtype=np.float32)
wind_data = np.zeros(shape, dtype=np.float32)
hum_data = np.zeros(shape, dtype=np.float32)

for t_idx in range(len(times)):
    temp_data[t_idx] = base_temp + rng.uniform(-3, 3, size=lat_mesh.shape)
    pres_data[t_idx] = 1013.0 - rng.uniform(2, 18, size=lat_mesh.shape)
    wind_data[t_idx] = rng.uniform(4, 30, size=lat_mesh.shape)
    hum_data[t_idx] = rng.uniform(35, 95, size=lat_mesh.shape)

ds = xr.Dataset(
    {
        "temperature": (["time", "lat", "lon"], temp_data),
        "pressure": (["time", "lat", "lon"], pres_data),
        "wind_speed": (["time", "lat", "lon"], wind_data),
        "humidity": (["time", "lat", "lon"], hum_data),
    },
    coords={
        "time": times,
        "lat": lats,
        "lon": lons,
    },
)

ds.to_netcdf(nc_path)
print(f"Dataset generated at {nc_path}. File size: {os.path.getsize(nc_path)} bytes")

# Test opening
opened_ds = xr.open_dataset(nc_path)
print("Opened dataset successfully:")
print(opened_ds)
