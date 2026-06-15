// 01_export_feature_grid.js
// Google Earth Engine script for data collection.
// Exports Sentinel-2 + soil/climate/terrain feature points for Iizuka, Sosa, Chiba, Japan.
// Save exported CSV to: data/raw/sosa_crop_features_2025.csv

// -----------------------------------------------------------------------------
// 1. AOI
// -----------------------------------------------------------------------------
// Bounding box for Iizuka district, Sosa City, Chiba Prefecture, Japan.
var aoi = ee.Geometry.Polygon([
  [[140.535, 35.718], [140.569, 35.718], [140.569, 35.746], [140.535, 35.746], [140.535, 35.718]]
], null, false);

var START_DATE = '2025-01-01';
var END_DATE = '2025-12-31';
var EXPORT_SCALE = 10; // 10m Sentinel-2 native resolution (highly detailed for small village scale)
var MAX_POINTS = 50000; // sample density for the 10 km2 area
var EXPORT_FOLDER = 'japan_crop_suitability_poc';

Map.centerObject(aoi, 14);
Map.addLayer(aoi, {color: 'white'}, 'AOI', false);

// -----------------------------------------------------------------------------
// 2. Sentinel-2 preprocessing
// -----------------------------------------------------------------------------
function renameAndScaleS2(img) {
  var scaled = img.select(
    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'SCL'],
    ['BLUE', 'GREEN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2', 'SCL']
  );

  var optical = scaled.select(['BLUE', 'GREEN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2'])
    .divide(10000);
  return optical.addBands(scaled.select('SCL')).copyProperties(img, img.propertyNames());
}

function maskS2Clouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))      // medium cloud probability
    .and(scl.neq(9))      // high cloud probability
    .and(scl.neq(10))     // cirrus
    .and(scl.neq(11));    // snow/ice
  return img.updateMask(mask);
}

function addSpectralIndices(img) {
  var blue = img.select('BLUE');
  var green = img.select('GREEN');
  var red = img.select('RED');
  var nir = img.select('NIR');
  var swir1 = img.select('SWIR1');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var evi = nir.subtract(red).multiply(2.5)
    .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename('EVI');
  var savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename('SAVI');
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI');
  var ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI');
  var bsi = swir1.add(red).subtract(nir.add(blue))
    .divide(swir1.add(red).add(nir).add(blue)).rename('BSI');

  return img.addBands([ndvi, evi, savi, ndwi, ndmi, ndbi, bsi]);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(renameAndScaleS2)
  .map(maskS2Clouds)
  .map(addSpectralIndices);

var s2Composite = s2.median().clip(aoi);

// -----------------------------------------------------------------------------
// 3. Soil, terrain, and climate proxy layers
// -----------------------------------------------------------------------------
// OpenLandMap pH is often stored as pH x 10.
var soilPh = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0').divide(10).rename('soil_ph').clip(aoi);

// Organic carbon proxy. Calibrate units before final use.
var soilOrganicCarbon = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0').rename('soil_org_carbon').clip(aoi);

var dem = ee.Image('USGS/SRTMGL1_003').clip(aoi);
var elevation = dem.rename('elevation_m');
var slope = ee.Terrain.slope(dem).rename('slope_deg');

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterBounds(aoi)
  .filterDate(START_DATE, END_DATE);
var meanTempC = era5.select('temperature_2m').mean().subtract(273.15).rename('mean_temp_c').clip(aoi);

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(aoi)
  .filterDate(START_DATE, END_DATE);
var annualRain = chirps.select('precipitation').sum().rename('annual_rain_mm').clip(aoi);

var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('worldcover_label').clip(aoi);

// Eligible land: tree, shrubland, grassland, cropland, bare/sparse.
var eligibleLand = worldcover.eq(10)
  .or(worldcover.eq(20))
  .or(worldcover.eq(30))
  .or(worldcover.eq(40))
  .or(worldcover.eq(60))
  .rename('eligible_land');

var featureStack = s2Composite.select([
    'BLUE', 'GREEN', 'RED', 'NIR', 'RE1', 'RE2', 'RE3', 'SWIR1', 'SWIR2',
    'NDVI', 'EVI', 'SAVI', 'NDWI', 'NDMI', 'NDBI', 'BSI'
  ])
  .addBands([soilPh, soilOrganicCarbon, slope, elevation, meanTempC, annualRain, worldcover, eligibleLand])
  .clip(aoi);

// -----------------------------------------------------------------------------
// 4. Visualization
// -----------------------------------------------------------------------------
Map.addLayer(s2Composite.select(['SWIR1', 'NIR', 'RED']), {min: 0, max: 0.35}, 'S2 false color', false);
Map.addLayer(s2Composite.select('NDVI'), {min: 0, max: 0.7, palette: ['brown', 'yellow', 'green']}, 'NDVI', false);
Map.addLayer(s2Composite.select('NDMI'), {min: -0.4, max: 0.5, palette: ['brown', 'yellow', 'blue']}, 'NDMI', false);
Map.addLayer(eligibleLand.updateMask(eligibleLand), {palette: ['00ff00']}, 'Eligible land', false);

// -----------------------------------------------------------------------------
// 5. Export as CSV points
// -----------------------------------------------------------------------------
// Use sample rather than full image export to keep the PoC lightweight.
var sampled = featureStack.sample({
  region: aoi,
  scale: EXPORT_SCALE,
  numPixels: MAX_POINTS,
  seed: 42,
  geometries: true,
  dropNulls: true
});

// Add longitude/latitude columns for Python dashboard.
var sampledWithLatLon = sampled.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    longitude: coords.get(0),
    latitude: coords.get(1)
  });
});

//print('Feature sample preview', sampledWithLatLon.limit(5));

Export.table.toDrive({
  collection: sampledWithLatLon,
  description: 'sosa_crop_features_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sosa_crop_features_2025',
  fileFormat: 'CSV'
});
