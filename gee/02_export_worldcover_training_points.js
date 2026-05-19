// 02_export_worldcover_training_points.js
// Optional script: export stratified WorldCover training points with Sentinel-2 features.
// Useful if you want a separate land-cover model like the UNICEF PoC.

var aoi = ee.Geometry.Polygon([
  [[36.05, 33.25], [36.95, 33.25], [37.05, 34.85], [36.15, 34.85], [36.05, 33.25]]
], null, false);

var START_DATE = '2025-01-01';
var END_DATE = '2025-12-31';
var EXPORT_FOLDER = 'syria_crop_suitability_poc';

function renameAndScaleS2(img) {
  var scaled = img.select(
    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'SCL'],
    ['BLUE', 'GREEN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2', 'SCL']
  );
  var optical = scaled.select(['BLUE', 'GREEN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']).divide(10000);
  return optical.addBands(scaled.select('SCL')).copyProperties(img, img.propertyNames());
}

function maskS2Clouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask);
}

function addSpectralIndices(img) {
  var blue = img.select('BLUE');
  var green = img.select('GREEN');
  var red = img.select('RED');
  var nir = img.select('NIR');
  var swir1 = img.select('SWIR1');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename('EVI');
  var savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename('SAVI');
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI');
  var ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI');
  var bsi = swir1.add(red).subtract(nir.add(blue)).divide(swir1.add(red).add(nir).add(blue)).rename('BSI');
  return img.addBands([ndvi, evi, savi, ndwi, ndmi, ndbi, bsi]);
}

var s2Composite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(renameAndScaleS2)
  .map(maskS2Clouds)
  .map(addSpectralIndices)
  .median()
  .clip(aoi);

var featureBands = [
  'BLUE', 'GREEN', 'RED', 'NIR', 'RE1', 'RE2', 'RE3', 'SWIR1', 'SWIR2',
  'NDVI', 'EVI', 'SAVI', 'NDWI', 'NDMI', 'NDBI', 'BSI'
];

var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('label').clip(aoi);
var trainingImage = s2Composite.select(featureBands).addBands(worldcover);

var samples = trainingImage.stratifiedSample({
  numPoints: 800,
  classBand: 'label',
  region: aoi,
  scale: 10,
  seed: 42,
  geometries: true
});

var samplesWithLatLon = samples.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({longitude: coords.get(0), latitude: coords.get(1)});
});

print('WorldCover training samples', samplesWithLatLon.limit(5));

Export.table.toDrive({
  collection: samplesWithLatLon,
  description: 'syria_worldcover_training_points_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'syria_worldcover_training_points_2025',
  fileFormat: 'CSV'
});
