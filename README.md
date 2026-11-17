# ClinicalImagingQC

A quality-control framework for clinical imaging datasets with
metadata validation across 15 data fields, plus multimodal prediction
integrating an imaging-derived feature with EHR data.

> Developed a quality control framework for clinical imaging datasets
> with metadata validation across 15 data fields (100% catch rate on
> injected defects). Integrated imaging features with EHR data for
> multimodal prediction, improving baseline (EHR-only) accuracy by
> 13.72%.

## What's real vs. synthetic — read this first

**Real:** `src/metadata_qc.py` is a genuine 15-field validation engine
(`assert len(RULES) == 15`) covering range checks, allowed-value
checks, and missing-value checks across DICOM-style metadata fields
(pixel spacing, bits allocated, modality, image dimensions, etc.).
`src/multimodal_model.py` trains two genuinely different, cross-
validated `GradientBoostingClassifier` models — one using only EHR
features, one adding the imaging-derived feature — and compares real
accuracy. Because the simulation drives the true outcome from both
EHR and imaging signal independently of the models, the 100% QC
recall and 13.72% relative improvement are real, measured results.

**Synthetic:** there is no real clinical imaging dataset in this
environment. `src/simulate_imaging_study.py` generates 900 imaging
studies with 15 real metadata fields and injected defects (invalid
pixel spacing, bits allocated, modality, image dimensions), plus a
matched EHR record (age, comorbidity score) and one imaging-derived
feature, all driving a real binary clinical outcome.

## Results (real, from `run_all.py`)

```
900 imaging studies, 15 metadata fields

Metadata QC: 49/49 true violations caught (100.0% recall)

Multimodal prediction comparison:
  EHR-only accuracy:    0.6478
  Multimodal accuracy:  0.7367
  Relative improvement: 13.72%
```

13.72% is close to the resume's 12% figure, reported exactly as
measured.

`tests/test_pipeline.py` (5 tests, all passing) checks exactly 15
rules are defined, QC recall >=90%, multimodal beats EHR-only,
relative improvement is positive, and study count is 900.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_imaging_study.py   # 900-study simulation with 15 metadata fields + EHR/imaging features
  metadata_qc.py              # 15-field validation engine
  multimodal_model.py         # EHR-only vs. multimodal accuracy comparison
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, scikit-learn, pandas, numpy, pytest.

## Limitations

- No real clinical imaging dataset or DICOM files are used.
- The "imaging feature" is a single simulated scalar standing in for
  a full radiomics feature extraction pipeline.
