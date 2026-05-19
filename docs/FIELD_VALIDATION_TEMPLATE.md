# Field validation template

Create a CSV file:

```text
data/raw/field_validation.csv
```

Recommended columns:

```text
plot_id,crop,longitude,latitude,suitability_label,soil_ph_lab,organic_matter_lab,ec_salinity_lab,texture,irrigation_source,irrigation_frequency,current_crop,farmer_notes,photo_url
```

## suitability_label

| Label | Meaning |
|---:|---|
| 0 | clearly unsuitable |
| 1 | low suitability |
| 2 | moderate suitability |
| 3 | high suitability |
| 4 | excellent / priority trial zone |

## Minimum first validation for 0.1 ha farmer plot

- GPS polygon or centroid
- soil pH
- organic matter
- EC/salinity
- texture
- water source
- farmer assessment
- crop history
- photos

## How this changes the model

The current model is pseudo-supervised. With field validation labels, we can train a real supervised crop model using the same feature columns.
