# Datasets

GeoChemAD bundles **16 geochemical anomaly-detection tasks** built from public
government geoscience surveys. Each task is a 2-D point cloud of geochemical
samples (multi-element assays) plus a set of known mineral-deposit sites used
**only as evaluation labels** — no labels enter training.

All CSVs live under `data/gswa/`:

```
data/gswa/
├── geochemical/   # one row per sample: X, Y, sample metadata, element columns
└── site/          # one row per known deposit: X, Y (positive evaluation labels)
```

## The 16 areas

| area_id          | target  | medium     | geochemical CSV          | site CSV                      |
|------------------|---------|------------|--------------------------|-------------------------------|
| area1_sed1_au    | Au_ppm  | sediment   | area1_sediment_au.csv    | area1_sediment_au_site.csv    |
| area2_sed2_cu    | Cu_ppm  | sediment   | area2_sediment_cu.csv    | area2_sediment_cu_site.csv    |
| area3_rock1_w    | W_ppm   | rockchip   | area3_rockchip_w.csv     | area3_rockchip_w_site.csv     |
| area4_rock2_au   | Au_ppm  | rockchip   | area4_rockchip_au.csv    | area4_rockchip_au_site.csv    |
| area5_rock3_cu   | Cu_ppm  | rockchip   | area5_rockchip_cu.csv    | area5_rockchip_cu_site.csv    |
| area6_soil1_au   | Au_ppm  | soil       | area6_soil_au.csv        | area6_soil_au_site.csv        |
| area7_soil2_au   | Au_ppm  | soil       | area7_soil_au.csv        | area7_soil_au_site.csv        |
| area8_soil3_ni   | Ni_ppm  | soil       | area8_soil_ni.csv        | area8_soil_ni_site.csv        |
| area9_dh1_au     | Au_ppm  | drillhole  | area9_drillhole_au.csv   | area9_drillhole_au_site.csv   |
| area10_dh2_ag    | Ag_ppm  | drillhole  | area10_drillhole_au.csv  | area10_drillhole_ag_site.csv  |
| area11_geo1_au   | Au_ppm  | geochem    | area11-14_geochem.csv    | area11_geochem_au_site.csv    |
| area12_geo2_cu   | Cu_ppm  | geochem    | area11-14_geochem.csv    | area12_geochem_cu_site.csv    |
| area13_geo3_ni   | Ni_ppm  | geochem    | area11-14_geochem.csv    | area13_geochem_ni_site.csv    |
| area14_geo4_w    | W_ppm   | geochem    | area11-14_geochem.csv    | area14_geochem_w_site.csv     |
| area15_sed3_au   | AU      | sediment   | area15-16_geochem.csv    | area15_geochem_au_site.csv    |
| area16_sed4_cu   | CU      | sediment   | area15-16_geochem.csv    | area16_geochem_cu_site.csv    |

Notes:
- **areas 11–14** share one multi-element CSV (`area11-14_geochem.csv`), each
  defining a different target element and its own deposit-site labels.
- **areas 15–16** are a Canadian survey with **uppercase oxide/element column
  naming** (`AU`, `CU`, `SIO2`, …) and no `_ppm` suffix — handled by a separate
  element list in [`shared/area_config.py`](../shared/area_config.py).
- The canonical definition of every task (paths, target, element lists,
  per-area hyper-parameters) is `shared/area_config.py`. **Edit nothing here by
  hand** — change the config.

## Column conventions

- Coordinate columns: `X`, `Y` (projected metres). Metadata columns excluded
  from features: `X, Y, SAMPLEID, SAMPLETYPE, WAMEX_A_NO, COMPSAMPID,
  Longitude, Latitude`.
- Element columns carry unit suffixes `_ppm`, `_ppb`, `_pct`. The element
  filter regex is `^[A-Za-z]+_(ppm|ppb|pct)$`; **oxide forms** (e.g.
  `Al2O3_pct`) are redundant with their elemental equivalents and dropped.
- **Missing / below-detection-limit (BDL) sentinel: `-9999`.** Handling is
  described in [configuration.md](configuration.md#preprocessing).

## Data root override

By default the code reads the bundled `data/gswa/`. To point at an external
copy without editing code:

```bash
export GEOCHEM_DATA_ROOT=/path/to/your/data/gswa
```

`shared/area_config.py` resolves `GEO_DIR`/`SITE_DIR` from this variable,
falling back to the in-repo `data/gswa/`.

## Provenance & attribution

Geochemical assays and deposit sites are derived from open government
geoscience data — the **Geological Survey of Western Australia (GSWA / WAMEX)**
for areas 1–14 and a **public Canadian regional geochemical survey** for
areas 15–16. They are redistributed for research reproducibility and remain
subject to the original providers' terms; cite the original surveys alongside
this repository (see [LICENSE](../LICENSE)).
