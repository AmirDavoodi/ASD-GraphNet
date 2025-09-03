# ASD-GraphNet

ASD-GraphNet is a graph-learning framework for Autism Spectrum Disorder (ASD) diagnosis using resting-state fMRI data. It builds functional brain networks from ABIDE I, extracts node-, edge-, and graph-level features, applies feature engineering (Mutual Information, PCA), and evaluates classical and ensemble classifiers (e.g., SVM, Logistic Regression, XGBoost, LightGBM). The best model achieved 75.25% accuracy on ABIDE I.

Article title: ASD-GraphNet: A novel graph learning approach for Autism Spectrum Disorder diagnosis using fMRI data.

Scientific article: see the published paper on [Computers in Biology and Medicine](https://doi.org/10.1016/j.compbiomed.2025.110723).

## Highlights
- ASD-GraphNet: Novel graph-based framework for ASD diagnosis via fMRI.
- 75.25% accuracy, beats SOTA by 2–5% on ABIDE I dataset.
- Uses Craddock 200, AAL, Dosenbach 160 atlases for connectivity.
- Interpretable pipeline for neuroimaging and early ASD detection.
- Code: https://github.com/AmirDavoodi/ASD-GraphNet.

## Repository structure
- `download.py`: Downloader for ABIDE I ROI time-series files via ABIDE public S3
- `utils/`: Core feature extraction and analysis modules
  - `graph_generation.py`: Builds NetworkX graphs per subject from ROI time-series
  - `graph_features.py`: Computes node-, edge-, and graph-level features
  - `shap_analysis.py`: SHAP-based interpretability utilities
- `experiments/`: Scripts and notebooks for pipelines and SHAP analyses
- `outputs/`: Example outputs (figures, CSVs) to help verify reproduction
- `requirements.txt`: Python dependencies

## Environment setup
- Python >= 3.10 recommended
- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Data acquisition (ABIDE I)
This repository expects ABIDE I ROI time-series files for atlases CC200, AAL, and Dosenbach 160. You will need the ABIDE I phenotype CSV and will download ROI time-series from the public S3.

1) Obtain the phenotype file `pheno_file.csv` from NITRC (ABIDE I) and place it at the repository root.
2) Use the helper downloader to fetch time-series into your chosen output folders (folders will be created if missing):
```bash
python download.py pheno_file.csv atlases/dos160 --roi rois_dosenbach160
python download.py pheno_file.csv atlases/aal    --roi rois_aal
python download.py pheno_file.csv atlases/cc200  --roi rois_cc200
```
Notes:
- Default pipeline is `cpac` and filter is `filt_global` (see `download.py` for options)
- Ensure stable network connection; the downloader skips missing files and continues

## Quickstart (feature extraction via notebooks)
Launch Jupyter and run the notebooks in order:
```bash
jupyter notebook
```
Recommended flow:
1) `graph_featrues_extraction.ipynb` (extract subject-wise graphs and features)
2) `experiments_pipeline.ipynb` (feature engineering + modeling)
3) `experiments_pipeline_classifiers.ipynb` or `experiments_pipeline_classifiers_revised.ipynb` (extended model comparisons)
4) `experiments_Dataset_Visual.ipynb` (visualizations)

## Reproducing main results (scripts)
Several steps have script equivalents under `experiments/`:
- `calculate_balancing_performance.py`: Site balancing performance analysis
- `run_shap_analysis.py`: End-to-end SHAP computation and plots
- `shap_*` modules: Training, data loader, visualization utilities

Example (SHAP end-to-end):
```bash
python experiments/run_shap_analysis.py
```
Outputs will be written under `outputs/` (e.g., `outputs/shap_analysis/...`).

## Outputs
This repository includes example outputs for reference and sanity-checking, such as adjacency matrices, graph figures, and feature importance CSVs under `outputs/`. Your exact values may differ slightly due to randomness or environment differences.

## Interpretability
Global and local interpretability are provided via SHAP. See `experiments/run_shap_analysis.py` or the SHAP notebooks for:
- Global feature importance bar and beeswarm plots
- Dependence plots for top features
- Individual prediction explanations (waterfall/force plots)

## Citation
If you use this code or derive ideas from it, please cite the published article and this repository. Example BibTeX:
```bibtex
@article{ZERAATI2025110723,
  title = {ASD-GraphNet: A novel graph learning approach for Autism Spectrum Disorder diagnosis using fMRI data},
  journal = {Computers in Biology and Medicine},
  volume = {196},
  pages = {110723},
  year = {2025},
  issn = {0010-4825},
  doi = {https://doi.org/10.1016/j.compbiomed.2025.110723},
  url = {https://www.sciencedirect.com/science/article/pii/S0010482525010741},
  author = {Mina Zeraati and Amirehsan Davoodi},
  keywords = {Autism Spectrum Disorder (ASD), fMRI, Graph machine learning, Brain connectivity networks, Brain atlases, Functional connectivity networks, ABIDE dataset},
  abstract = {Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition with heterogeneous symptomatology, making accurate diagnosis challenging. Traditional methods rely on subjective behavioral assessments, often overlooking subtle neural biomarkers. This study introduces ASD-GraphNet, a novel graph-based learning framework for diagnosing ASD using functional Magnetic Resonance Imaging (fMRI) data. Leveraging the Autism Brain Imaging Data Exchange (ABIDE) dataset, ASD-GraphNet constructs brain networks based on established atlases (Craddock 200, AAL, and Dosenbach 160) to capture intricate connectivity patterns. The framework employs systematic preprocessing, graph construction, and advanced feature extraction to derive node-level, edge-level, and graph-level metrics. Feature engineering techniques, including Mutual Information-based selection and Principal Component Analysis (PCA), are applied to enhance classification performance. ASD-GraphNet evaluates a range of classifiers, including Logistic Regression, Support Vector Machines, and ensemble methods like XGBoost and LightGBM, achieving an accuracy of 75.25% in distinguishing individuals with ASD from healthy controls. This demonstrates the framework’s potential to provide objective, data-driven diagnostics based solely on resting-state fMRI data. By integrating graph-based learning with neuroimaging and addressing dataset imbalance, ASD-GraphNet offers a scalable and interpretable solution for early ASD detection, paving the way for more reliable interventions. The GitHub repository for this project is available at: https://github.com/AmirDavoodi/ASD-GraphNet.}
}
```

## License
See `LICENSE` for terms. Code is provided for research and educational use.

## Contact
- For research questions: open an issue on GitHub
- Mina Zeraati: minazeraatii@gmail.com
- Amirehsan Davoodi: amirehsan.davoodi@gmail.com