# LLM-Selected Pathfinder Features — GeoChemADv2

> **Feature Selection Method**: LLM (Claude) geological knowledge-guided selection.
> Features are chosen based on established geochemical pathfinder theory for each inferred deposit type, constrained to columns available in the dataset.
> This file serves as the **backbone feature specification** for all models (T1, T2, VAE, baselines).
> Results obtained using these features should be labelled **"LLM-FS"** in comparison tables.

---

## Principles

- **Target element** is always included as a feature (anomaly detection is unsupervised; high target values are not labels but part of the geochemical signature).
- **Oxide-ppm redundancy**: when both forms exist (e.g. `Al_ppm` and `Al2O3_pct`), only the `_pct` form is listed for major elements; both are redundant after CLR.
- **Missing columns**: if a listed column is absent or 100% BDL in a given area, it is silently skipped during loading.
- **Non-element metadata** (`OBJECTID*`, `GSWANO`, `EXTRACT_DA`, `SITENO`) are always excluded regardless of this list.

---

## Per-Area Specifications

---

### area1 — `area1_sed1_au`
| Field | Value |
|---|---|
| Sample type | Sediment |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au (Yilgarn Craton, WA) |

**Reasoning**: Katanning area sediment survey. Orogenic Au systems in the Yilgarn are characterised by a As-Sb-Bi-Te-Ag suite in hydrothermal fluids, with Hg and Tl as distal pathfinders. W and Mo indicate proximal granite-related fluids.

**Input features**:
```
Au_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Ag_ppm, Hg_ppm, Tl_ppm,
W_ppm, Mo_ppm, Se_ppm, Cu_ppm, Pb_ppm, Zn_ppm, Co_ppm, Ni_ppm,
Fe2O3T_pct, MnO_pct
```

---

### area2 — `area2_sed2_cu`
| Field | Value |
|---|---|
| Sample type | Sediment |
| Target | `Cu_ppm` |
| Inferred deposit type | Porphyry Cu-Au or VHMS |

**Reasoning**: Cu-targeted sediment survey. Porphyry systems show a Cu-Mo-Au-Re core with Pb-Zn-As-Sb zoning outward. VHMS adds Ba and high Zn/Pb. S and Se indicate sulfide saturation.

**Input features**:
```
Cu_ppm, Au_ppm, Ag_ppm, Mo_ppm, Re_ppm, Zn_ppm, Pb_ppm, Co_ppm,
Ni_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Se_ppm, S_pct,
Fe2O3T_pct, MnO_pct, Al2O3_pct
```

---

### area3 — `area3_rock1_w`
| Field | Value |
|---|---|
| Sample type | Rockchip |
| Target | `W_ppm` |
| Inferred deposit type | Skarn W or Greisen W-Sn |

**Reasoning**: W deposits occur in two main settings: (1) skarn — Ca-Fe-Mn alteration halos with Cu, Bi, Mo; (2) greisen — granite-hosted with Sn, Li, Rb, Cs, Be, F. Both settings show elevated As and elevated LOI (alteration). Including both suites covers uncertainty in deposit type.

**Input features**:
```
W_ppm, Sn_ppm, Mo_ppm, Bi_ppm, Cu_ppm, As_ppm, Pb_ppm, Zn_ppm,
F_ppm, Li_ppm, Rb_ppm, Cs_ppm, Be_ppm,
CaO_pct, Fe2O3T_pct, MnO_pct, Al2O3_pct, K2O_pct, SiO2_pct,
LOI_pct
```

---

### area4 — `area4_rock2_au`
| Field | Value |
|---|---|
| Sample type | Rockchip |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au (Yilgarn, WA) — **corrected from Li** |

**Reasoning**: Rockchip Au in a cratonic setting most likely represents orogenic lode gold. The classic As-Sb-Bi-Te-Ag-W-Mo suite applies. S indicates sulfide mineralisation; Fe reflects iron alteration (silica-carbonate-pyrite assemblage).

**Input features**:
```
Au_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Ag_ppm, Hg_ppm, Tl_ppm,
W_ppm, Mo_ppm, Se_ppm, S_pct, Cu_ppm, Pb_ppm, Zn_ppm,
Fe2O3T_pct, Al2O3_pct, K2O_pct, SiO2_pct
```

---

### area5 — `area5_rock3_cu`
| Field | Value |
|---|---|
| Sample type | Rockchip |
| Target | `Cu_ppm` |
| Inferred deposit type | Magmatic Cu-Ni-PGE (komatiite-hosted) |

**Reasoning**: Presence of Ir, Pd, Pt, Rh, Ru in this dataset strongly suggests a komatiite-hosted magmatic sulfide system. The key geochemical vectors are Ni-Co-Cr-Mg (ultramafic host), PGEs (magmatic sulfide indicator), and S-Se-Te (chalcophile suite). Cu and Au are co-products.

**Input features**:
```
Cu_ppm, Ni_ppm, Co_ppm, Au_ppm, Ag_ppm,
Pd_ppb, Pt_ppb, Ir_ppb, Rh_ppb, Ru_ppb,
S_pct, Se_ppm, Te_ppm,
MgO_pct, Fe2O3T_pct, Cr2O3_ppm, TiO2_pct, V_ppm, MnO_pct
```

---

### area6 — `area6_soil1_au`
| Field | Value |
|---|---|
| Sample type | Soil |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au — **limited panel** |

**Reasoning**: This dataset has only 12 numeric columns. All non-target non-metadata columns are included as they represent the full available pathfinder suite: Ag (classic Au pathfinder), Cu/Pb/Zn (base-metal halo), Co/Ni (mafic host indicator), Pd (magmatic component).

**Input features** (all usable columns):
```
Au_ppm, Ag_ppm, Co_ppm, Cu_ppm, Ni_ppm, Pb_ppm, Pd_ppb, Zn_ppm
```
*(ZnO_ppm excluded as redundant with Zn_ppm)*

---

### area7 — `area7_soil2_au`
| Field | Value |
|---|---|
| Sample type | Soil |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au |

**Reasoning**: Broader soil panel. As and Sb are the most reliable distal Au pathfinders in soil; Te, Hg, Tl indicate proximal hydrothermal fluids. W and Mo suggest granite-related or intrusion-related Au. Base-metal halo (Cu, Pb, Zn) and Fe-Mn for redox context.

**Input features**:
```
Au_ppm, As_ppm, Sb_ppm, Te_ppm, Ag_ppm, Hg_ppm, Tl_ppm,
W_ppm, Mo_ppm, Se_ppm, Cu_ppm, Pb_ppm, Zn_ppm,
Co_ppm, Ni_ppm, Fe2O3T_pct, MnO_pct
```

---

### area8 — `area8_soil3_ni`
| Field | Value |
|---|---|
| Sample type | Soil |
| Target | `Ni_ppm` |
| Inferred deposit type | Komatiite-hosted Ni-Cu-PGE |

**Reasoning**: PGEs (Pd, Pt, Ir) in a Ni-targeted soil survey clearly indicate komatiite-hosted sulfide mineralisation. Co, Cr, Mg, Fe, V reflect the ultramafic host. Cu is a co-product. S and Se are chalcophile pathfinders. Au is a minor co-product in many komatiite Ni deposits (e.g. Kambalda-style).

**Input features**:
```
Ni_ppm, Co_ppm, Cu_ppm, Au_ppm, Ag_ppm,
Pd_ppb, Pt_ppb, Ir_ppb,
S_pct, Se_ppm, Te_ppm,
MgO_pct, Fe2O3T_pct, Cr2O3_ppm, V_ppm, MnO_pct, TiO2_pct
```

---

### area9 — `area9_dh1_au`
| Field | Value |
|---|---|
| Sample type | Drillhole |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au or Intrusion-Related Au (IRGD) |

**Reasoning**: Drillhole data gives a 3-D view of the hydrothermal system. Orogenic Au: As-Sb-Bi-Te-Ag core. IRGD (Intrusion-Related Gold Deposit): Bi-Te-W-Mo dominant with lower As/Sb. Both suites are included. S and LOI reflect sulfidation and carbonation alteration respectively.

**Input features**:
```
Au_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Ag_ppm, Hg_ppm, Tl_ppm,
W_ppm, Mo_ppm, Se_ppm, S_pct, Cu_ppm, Pb_ppm, Zn_ppm,
Fe2O3T_pct, Al2O3_pct, K2O_pct, SiO2_pct, LOI_pct
```

---

### area10 — `area10_dh2_ag`
| Field | Value |
|---|---|
| Sample type | Drillhole |
| Target | `Ag_ppm` |
| Inferred deposit type | Epithermal Ag-Au or VHMS |

**Reasoning**: Epithermal Ag systems show Ag-Au-Pb-Zn-Cd-Sb-As-Bi-Te mineralisation with Mn-Fe alteration halos. VHMS adds Cu-Zn-Ba. Cd is a strong VHMS indicator (elevated Zn/Cd ratios). S indicates sulfide abundance. Ir presence suggests a possible deep magmatic component.

**Input features**:
```
Ag_ppm, Au_ppm, Cu_ppm, Pb_ppm, Zn_ppm, Cd_ppm,
As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Tl_ppm, Se_ppm,
S_pct, MnO_pct, Fe2O3T_pct, Ir_ppb
```

---

### area11 — `area11_geo1_au`
| Field | Value |
|---|---|
| Sample type | Multi-method geochem |
| Target | `Au_ppm` |
| Inferred deposit type | Orogenic Au |

**Reasoning**: Mixed geochem survey. Orogenic Au pathfinder suite applies. LOI and major oxides capture alteration intensity.

**Input features**:
```
Au_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Ag_ppm, Hg_ppm, Tl_ppm,
W_ppm, Mo_ppm, Se_ppm, S_pct, Cu_ppm, Pb_ppm, Zn_ppm,
Fe2O3T_pct, Al2O3_pct, K2O_pct, SiO2_pct, LOI_pct
```

---

### area12 — `area12_geo2_cu`
| Field | Value |
|---|---|
| Sample type | Multi-method geochem |
| Target | `Cu_ppm` |
| Inferred deposit type | Porphyry Cu-Au or VHMS |

**Input features**:
```
Cu_ppm, Au_ppm, Ag_ppm, Mo_ppm, Re_ppm, Zn_ppm, Pb_ppm,
Co_ppm, Ni_ppm, As_ppm, Sb_ppm, Bi_ppm, Te_ppm, Se_ppm, S_pct,
Fe2O3T_pct, MnO_pct, Al2O3_pct, SiO2_pct
```

---

### area13 — `area13_geo3_ni`
| Field | Value |
|---|---|
| Sample type | Multi-method geochem |
| Target | `Ni_ppm` |
| Inferred deposit type | Komatiite Ni-Cu-PGE or Laterite Ni |

**Reasoning**: Co, Cr, Mg, Fe, V reflect ultramafic host. PGEs (Pd, Pt) distinguish primary sulfide from laterite. Cu and Au are sulfide co-products. S and Se are chalcophile indicators.

**Input features**:
```
Ni_ppm, Co_ppm, Cu_ppm, Au_ppm, Ag_ppm,
Pd_ppb, Pt_ppb, Se_ppm, S_pct, Te_ppm,
MgO_pct, Fe2O3T_pct, Cr2O3_ppm, V_ppm, MnO_pct, TiO2_pct, Al2O3_pct
```

---

### area14 — `area14_geo4_w`
| Field | Value |
|---|---|
| Sample type | Multi-method geochem |
| Target | `W_ppm` |
| Inferred deposit type | Skarn W or Greisen W-Sn |

**Input features**:
```
W_ppm, Sn_ppm, Mo_ppm, Bi_ppm, Cu_ppm, As_ppm, Pb_ppm, Zn_ppm,
F_ppm, Li_ppm, Rb_ppm, Cs_ppm, Be_ppm,
CaO_pct, Fe2O3T_pct, MnO_pct, Al2O3_pct, K2O_pct, SiO2_pct, LOI_pct
```

---

### area15 — `area15_sed3_au`
| Field | Value |
|---|---|
| Sample type | Sediment (Canada) |
| Target | `AU` |
| Inferred deposit type | Orogenic Au or IRGD (Canadian Shield) |

**Reasoning**: Canadian sediment geochemical survey. Column names differ (no `_ppm` suffix). Orogenic/IRGD Au pathfinder suite: As-Sb-Bi-Te-Ag-Hg-Tl-W-Mo. Major oxides capture lithological context. S indicates sulfide. B and F indicate hydrothermal fluids.

**Input features** (Canada column names):
```
AU, AS_O, SB, BI, TE, AG, HG, TL,
W, MO, SE, S, CU, PB, ZN, CO, NI,
FE2O3_T, MNO, AL2O3, SIO2, K2O, NA2O
```

---

### area16 — `area16_sed4_cu`
| Field | Value |
|---|---|
| Sample type | Sediment (Canada) |
| Target | `CU` |
| Inferred deposit type | Porphyry Cu-Mo-Au or VHMS |

**Input features** (Canada column names):
```
CU, AU, AG, MO, RE, ZN, PB, CD,
CO, NI, AS_O, SB, BI, TE, SE, S,
FE2O3_T, MNO, AL2O3, SIO2
```

---

## Summary Table

| Area | Type | Target | Deposit Interpretation | # Features |
|------|------|--------|----------------------|-----------|
| area1_sed1_au | Sediment | Au | Orogenic Au (Yilgarn) | 18 |
| area2_sed2_cu | Sediment | Cu | Porphyry Cu-Au / VHMS | 18 |
| area3_rock1_w | Rockchip | W | Skarn W / Greisen W-Sn | 21 |
| area4_rock2_au | Rockchip | Au | Orogenic Au (Yilgarn) | 19 |
| area5_rock3_cu | Rockchip | Cu | Magmatic Cu-Ni-PGE (komatiite) | 20 |
| area6_soil1_au | Soil | Au | Orogenic Au — limited panel | 8 |
| area7_soil2_au | Soil | Au | Orogenic Au | 17 |
| area8_soil3_ni | Soil | Ni | Komatiite Ni-Cu-PGE | 17 |
| area9_dh1_au | Drillhole | Au | Orogenic Au / IRGD | 21 |
| area10_dh2_ag | Drillhole | Ag | Epithermal Ag-Au / VHMS | 16 |
| area11_geo1_au | Geochem | Au | Orogenic Au | 21 |
| area12_geo2_cu | Geochem | Cu | Porphyry Cu-Au / VHMS | 19 |
| area13_geo3_ni | Geochem | Ni | Komatiite Ni-Cu-PGE / Laterite | 17 |
| area14_geo4_w | Geochem | W | Skarn W / Greisen W-Sn | 21 |
| area15_sed3_au | Sediment (CA) | Au | Orogenic Au / IRGD | 22 |
| area16_sed4_cu | Sediment (CA) | Cu | Porphyry Cu-Mo / VHMS | 20 |
