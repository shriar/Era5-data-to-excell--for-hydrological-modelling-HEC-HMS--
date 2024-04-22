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
- xlsxwriter

# Method: Reprojecting the layer(shape file) (from UTM to WGS)
1. Open your QGIS project and add the polygon shapefile.
2. Right-click on the layer in the Layer list and select "Set Layer CRS."
3. In the "Set Layer CRS" window, search for "UTM 46N" or type the EPSG code for the specific UTM 46N projection you are using (e.g., EPSG:32746).
4. Make sure the selected CRS matches the actual projection of your shapefile.
5. Click "OK" to apply the CRS.
6. Right-click on the layer again and select "Save As..."
7. In the "Save As" window, select a new file name and location for the converted shapefile.
8. Under "CRS," choose "WGS84" (EPSG:4326).
9. Uncheck the "Add layer to map" option to avoid duplicate layers.
10. Click "Save" to create the new shapefile in WGS84.