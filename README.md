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

### Hardware
All experiments were conducted on a Linux server running Ubuntu OS, equipped with an AMD Ryzen 9 5900HX processor (3.30 GHz), 39 GB of RAM, and an NVIDIA GeForce RTX 3060 GPU with 3840 CUDA cores and 6GB of GPU memory. We utilized libraries such as CUDA, PyTorch, scikit-learn, XGBoost, and LightGBM for our experiments.

### Software
- Python >= 3.10 recommended
  - For some Python installations you may also need to install the package `python-venv` in addition to standard Python. For example, a Linux Mint (a Ubuntu / Debian - derivative) installation requires:
  ```bash
   sudo apt install python3.NN-venv 
  ```
  (where NN is your Python version)
  - Note that you may need to invoke Python with `python3` rather than `python`
  - pip (pip3) may also need installing at this point, Debian and derivatives as example `sudo apt install python3-pip`. Similar to invoking Python you may need to call `pip3` rather than `pip`

- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Data acquisition (ABIDE I)
This repository expects [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) ROI time-series files for atlases CC200, AAL, and Dosenbach 160. You will need the ABIDE I phenotype CSV and will download ROI time-series from the public S3.

1) Obtain the phenotype file `pheno_file.csv` from NITRC (ABIDE I) and place it at the repository root.

2) Use the helper downloader (`download.py`) to fetch the ROI time-series datasets into your chosen output folders (folders will be created if missing):
```bash
python download.py pheno_file.csv atlases/dos160 --roi rois_dosenbach160
python download.py pheno_file.csv atlases/aal    --roi rois_aal
python download.py pheno_file.csv atlases/cc200  --roi rois_cc200
```
Notes:
- Default pipeline is `cpac` and filter is `filt_global` (see `download.py` for options)
- Ensure stable network connection; the downloader skips missing files and continues

3) The Nifty format data files (nii) were last downloaded in a gzip compressed format December 2025 and have the following sha256 checksums:
```bash
40b1edfa68bccf06f608661b79655726f59485498b665bd9bac7f86394d6b9c2 aal_roi_atlas.nii.gz
9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650 cc200_roi_atlas.nii.gz
c0bbd454a178006b42e725e0400705de64f752e9e82a0539ef5bfe10ae946020 dos160_roi_atlas.nii.gz
```

4) For convenience the file `nii_256checksums.txt` is provided and can be used to verify the content of the binary files e.g at a Linux command pronpt type : `$ sha256sum -c nii_256checksums.txt`. If you have the exact same version of data files used in this project you should see the following output:
```bash
aal_roi_atlas.nii.gz: OK
cc200_roi_atlas.nii.gz: OK
dos160_roi_atlas.nii.gz: OK
```

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
