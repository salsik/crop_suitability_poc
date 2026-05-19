// 03_export_farmer_plot_features.js
// Use this for the 0.1 hectare farmer trial plot.
// Replace farmerPlot with exact GPS polygon from the Syrian farmer.

var farmerPlot = ee.Geometry.Polygon([
  [[36.50, 34.00], [36.501, 34.00], [36.501, 34.001], [36.50, 34.001], [36.50, 34.00]]
], null, false);

var buffer = farmerPlot.buffer(200); // add context around the plot
var START_DATE = '2025-01-01';
var END_DATE = '2025-12-31';
var EXPORT_FOLDER = 'syria_crop_suitability_poc';

// For brevity, this script exports only plot geometry reference.
// Recommended: copy the featureStack logic from 01_export_feature_grid.js and replace `aoi` with `buffer`.
Map.centerObject(farmerPlot, 18);
Map.addLayer(farmerPlot, {color: 'red'}, 'Farmer 0.1 ha trial plot');
Map.addLayer(buffer, {color: 'yellow'}, '200m context buffer', false);

var feature = ee.Feature(farmerPlot, {
  plot_id: 'trial_plot_001',
  farmer_contact_status: 'agreed_for_trial',
  area_ha_approx: farmerPlot.area().divide(10000)
});

Export.table.toDrive({
  collection: ee.FeatureCollection([feature]),
  description: 'farmer_trial_plot_boundary',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'farmer_trial_plot_boundary',
  fileFormat: 'GeoJSON'
});
