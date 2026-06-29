# Changelog

All notable changes to MLens are documented here.

## [0.4.0] — Deep Learning & Concept Drift

### Added
- Native PyTorch support via `PyTorchExplainer` (DeepExplainer + GradientExplainer)
- Integrated Gradients support via Captum
- Concept drift detection: ADWIN, Page-Hinkley, DDM algorithms
- MLflow integration (`MLflowTracker`) — log audits as tracked runs
- Weights & Biases integration (`WandbTracker`)
- 19 new unit tests (`test_pytorch.py`, `test_concept_drift.py`)

### Changed
- `requirements.txt` updated with `torch`, `mlflow`, `wandb`

---

## [0.3.0] — Deploy Anywhere

### Added
- REST API via FastAPI (`api/main.py`)
  - `POST /audit` — full audit
  - `POST /audit/shap` — SHAP only
  - `POST /audit/fairness` — fairness only
  - `POST /audit/drift` — drift only
  - `GET /health`, `GET /version`
- Docker support: `docker/Dockerfile`, `docker/Dockerfile.dashboard`, `docker-compose.yml`
- GitHub Actions workflow for automated Docker image publishing
- 20 new API integration tests (`test_api.py`)

### Fixed
- CORS headers for cross-origin API calls
- Improved error messages for mismatched input shapes

---

## [0.2.0] — Production Ready

### Added
- CLI tool (`mlens audit ...`) via `mlens/cli/main.py`
- Streamlit live dashboard (`dashboard/app.py`)
- PDF report export via ReportLab (`pdf_generator.py`)
- `setup.py` for `pip install -e .` and console script registration
- 30 new unit tests across auditor, fairness, and drift modules

### Fixed
- KS-test crash on constant features
- SHAP dual-output handling for binary classifiers

---

## [0.1.0] — Initial Release

### Added
- Core `ModelAuditor` orchestrator
- SHAP explainability (`ShapAnalyzer`) — Tree/Linear/Kernel auto-selection
- Fairness evaluation (`FairnessEvaluator`) — demographic parity, equalized
  odds, disparate impact
- Drift detection (`DriftDetector`) — PSI + KS-test per feature
- HTML report generation
- Initial documentation and examples
