# Data Provenance and Offline Reference Bundle

## 1. Bundled Offline Reference Views (`data/reference/`)

To enable deterministic CPU verification and full simulation reproduction without
redistributing gigabytes of raw count matrices, `data/reference/` bundles the exact
annotation-blind derived baseline packages (`2.95 MB` compressed total):

- `B1_baseline.npz`: Human lymph node slice 1 (`RNA + ADT`, `3,484` spots, `174` tiles)
- `B2_baseline.npz`: Human tonsil slice 1 (`RNA + ADT`, `4,326` spots, `216` tiles)
- `B3_baseline.npz`: Mouse embryonic brain E15.5 (`MISAR-seq RNA + ATAC`, `1,949` spots, `97` tiles, `95` scorable)
- `B4_baseline.npz`: Mouse embryonic brain E18.5 (`MISAR-seq RNA + ATAC`, `2,129` spots, `106` tiles)

Each archive contains `P0` (`(n, 16)` MAD-scaled SVD RNA embedding), `Q0` (`(n, 16)`
paired-modality embedding), `coords` (`(n, 2)` spatial lattice coordinates), `ids`,
`R_static` (initial 8-cluster RNA partition), `I_bg` (initial 8-cluster paired
partition), `tiles`, and `tile_*` summary attributes. Notice that coordinates are
array spot-grid index units (`10x Visium` staggered lattice for `B1`/`B2` with
nearest-neighbor spacing `sqrt(2) = 1.4142`; `50x50 MISAR-seq` grid for `B3`/`B4`
with nearest-neighbor spacing `1.0`).

## 2. Public Upstream Accessions and Documented Provenance Gaps

Original matrices trace to public repositories:
- `B1`: SpatialGlue release (`Zenodo 10.5281/zenodo.10362607`).
- `B2`: GEO `GSE263617` (`GSM8195495`, `GSM8195499`), retaining `4,326` of `4,337`
  public in-tissue barcodes (`11` omitted upstream via an undocumented filter) and `31`
  ADT antibodies after excluding four mouse/rat IgG isotype controls.
- `B3` / `B4`: MISAR-seq (`NGDC OEP003285`, `NCBI PRJNA1081293`, `Zenodo 7714382`). On
  `B3`, local RNA counts match public raw matrix `E15_5-S1_raw_feature_bc_matrix.h5`
  identically (`max |diff| = 0`), whereas local ATAC contains `161,461` peaks vs.
  `141,420` in the public snapshot (`sum diff = 397,132`). On `B4`, upstream raw
  fragment-to-matrix scripts and donor IDs remain unreleased upstream.

All local benchmark scripts operate strictly on raw `X`, spot IDs, and spatial
coordinates via `preprocessing.py` and never read stored biological annotation labels.
Placing raw `.h5ad` files under `data/raw/{B1,B2,B3,B4}/` enables running optional
raw-count pipeline verification scripts (`scripts/run_count_level_check.py`).
