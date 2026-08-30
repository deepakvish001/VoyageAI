# 🚄 VoyageAI

<p align="center">
  <strong>Vande Bharat route travel-time prediction with scikit-learn and portable ONNX inference.</strong>
</p>

<p align="center">
  Python • pandas • scikit-learn • Gradient Boosting • ONNX Runtime
</p>

---

## Overview

VoyageAI trains a machine-learning baseline that predicts the total journey time of a Vande Bharat Express route from its distance. The project cleans a small route dataset, trains a Gradient Boosting regressor, evaluates it with Mean Absolute Error, exports the model to ONNX, and provides a lightweight inference command.

The current model uses one feature—route distance—so its output should be treated as a baseline rather than a timetable or operational forecast.

## Current Pipeline

\`\`\`text
vande_bharat.csv
       │
       ▼
Distance and travel-time parsing
       │
       ▼
Missing-row removal
       │
       ▼
Train/test split
       │
       ▼
GradientBoostingRegressor
       │
       ├── Mean Absolute Error evaluation
       └── ONNX export
                 │
                 ▼
        onnxruntime inference
\`\`\`

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | pandas and NumPy |
| Model | scikit-learn GradientBoostingRegressor |
| Evaluation | Mean Absolute Error |
| Model conversion | skl2onnx |
| Portable inference | ONNX Runtime |
| Dataset | Vande Bharat route data in CSV |

## Project Structure

\`\`\`text
VoyageAI/
├── README.md
├── .gitignore
└── VoyageAI/
    ├── train.py
    ├── inference.py
    ├── requirements.txt
    ├── vande_bharat.csv
    ├── vande_bharat_travel_time.onnx
    └── README.md
\`\`\`

## Dataset

The training script reads:

- Distance
- Travel Time

It converts distance values containing kilometres into floating-point values and travel-time strings containing hours and minutes into total minutes. Rows that cannot be parsed are removed.

The dataset contains approximately 42 usable routes after cleaning. This is too small for strong generalisation claims.

## Model

The baseline uses:

\`\`\`text
GradientBoostingRegressor
n_estimators = 100
learning_rate = 0.1
max_depth = 3
random_state = 42
\`\`\`

The data is split into training and test sets using a fixed random seed. Mean Absolute Error is reported in minutes.

## Complete Local Setup

### 1. Prerequisites

Install:

- Git
- Python 3.11 or newer
- pip
- A virtual-environment tool

Confirm your tools:

\`\`\`bash
python --version
pip --version
git --version
\`\`\`

### 2. Clone the repository

\`\`\`bash
git clone https://github.com/deepakvish001/VoyageAI.git
cd VoyageAI
\`\`\`

### 3. Create a virtual environment

Linux or macOS:

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
\`\`\`

Windows PowerShell:

\`\`\`powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
\`\`\`

### 4. Install dependencies

\`\`\`bash
pip install --upgrade pip
pip install -r VoyageAI/requirements.txt
\`\`\`

### 5. Train and export the model

The scripts currently resolve data and model paths relative to the working directory. Enter the inner project directory:

\`\`\`bash
cd VoyageAI
python train.py
\`\`\`

Successful training prints:

- parsed columns
- number of cleaned rows
- model-training completion
- Mean Absolute Error
- ONNX export confirmation

The exported file is:

\`\`\`text
vande_bharat_travel_time.onnx
\`\`\`

### 6. Run inference

From the inner VoyageAI directory:

\`\`\`bash
python inference.py 500
\`\`\`

The positional argument is route distance in kilometres. If omitted, the default is 500 km:

\`\`\`bash
python inference.py
\`\`\`

The command prints predicted travel time in minutes and hours.

## Reproducible Workflow

From the repository root:

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r VoyageAI/requirements.txt
cd VoyageAI
python train.py
python inference.py 500
\`\`\`

## Current Limitations

- Only route distance is used as a feature.
- The cleaned dataset is small.
- A single train/test split gives an unstable estimate.
- No cross-validation or uncertainty range is reported.
- Station count, stops, route type, region and timetable features are absent.
- Data lineage and update dates are not recorded.
- Parsing functions are embedded in the training script.
- File paths depend on the working directory.
- Hyperparameters are hard-coded.
- Model artifacts are committed without a metadata manifest.
- ONNX and scikit-learn prediction parity is not tested.
- Automated tests and continuous integration are absent.
- The model should not be used for passenger guarantees or railway operations.

## Validation Priorities

A stronger evaluation should include:

- repeated or cross-validated evaluation
- comparison with simple baselines
- Mean Absolute Error, Root Mean Squared Error and median error
- error by distance range
- residual inspection
- prediction sanity checks
- model-versus-ONNX parity
- reproducible data snapshots
- confidence or prediction intervals
- documented limitations

## Security and Data Guidance

- Treat external CSV content as untrusted input.
- Validate file size, columns, encodings and numeric ranges.
- Do not load untrusted pickle or joblib files.
- Pin and review dependencies.
- Keep API credentials and private datasets out of source control.
- Record model provenance and checksums.
- Avoid logging personal passenger information.
- Apply request limits if an inference API is introduced.

## Modernisation Roadmap

- Extract reusable parsing and feature modules
- Add structured configuration
- Add command-line options for paths and hyperparameters
- Introduce data validation
- Add baseline and model comparison
- Add cross-validation and experiment tracking
- Add model metadata and artifact checksums
- Verify ONNX prediction parity
- Add unit and integration tests
- Add linting, formatting and type checking
- Add CI
- Build a FastAPI inference service
- Add Docker support
- Add an interactive route dashboard
- Add drift and performance monitoring
- Expand route and timetable features
- Document a retraining and release process

## Contributing

Keep every pull request focused and independently verifiable.

\`\`\`bash
git checkout main
git pull --ff-only
git checkout -b feat/short-change-name
python -m venv .venv
source .venv/bin/activate
pip install -r VoyageAI/requirements.txt
git add .
git commit -m "feat: describe the change"
git push -u origin feat/short-change-name
\`\`\`

For model changes, include:

- data and feature changes
- baseline comparison
- evaluation metrics
- reproducibility details
- ONNX export impact
- limitations and rollback notes

## License

Add an explicit licence file before redistributing the dataset, model or source outside the intended project context.

---

<p align="center">
  A transparent baseline for exploring route travel-time prediction and portable ML inference.
</p>
