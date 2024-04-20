# Processing ERA5 data for HECHMS

- Basin shape file need to be corrected using GIS software before processing.
- change the grid size `self.grid_size = 0.25`, if needed.

```
self.geometries = self.gdf["geometry"]
self.subbasin_name = self.gdf["name"]  
```
This line may need to be changed according to your basin shape file. See the attribute table of you basin file for the correct subbasin name.


## Python Library need to be installed:
- xarray
- netCDF4
- numpy
- matplotlib
- shapely
- pandas
- geopandas