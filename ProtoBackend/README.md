# ProtoBackend - Prototypical Implicit Aspect Detection

ProtoBackend is a high-performance, offline module for performing **Implicit Attribute-Based Sentiment Analysis (ABSA)** using Prototypical Networks. It transforms labeled review sentences into a vector space of aspect centroids (prototypes) to enable zero-shot or few-shot inference.

## 🚀 Quick Start

Ensure you are using the project's virtual environment:
```powershell
# Run the complete episodic training & verification pipeline
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py run --dataset-family episodic
```

### Main Commands:
- `run` (Default): Executes the full pipeline (Train prototypes -> Sweep validation -> Calibrate thresholds -> Evaluate test set).
- `train`: Train prototypes only and save to disk.
- `eval`: Run evaluation on a specific split (`val` or `test`).
- `sweep`: Perform a hyperparameter sweep over different confidence thresholds and top-k values.
- `predict`: Run real-time inference on a single sentence.

---

## 📂 Project Structure

```text
ProtoBackend/
├── implicit_proto/  # Core logic (encoding, building prototypes, inference)
├── input/           # Source datasets (place your files here)
│   ├── episodic/    # Episodic training data (JSON/JSONL)
│   └── reviewlevel/ # Review-level training data (JSON/JSONL)
├── outputs/         # Generated artifacts, models, and evaluation reports
└── proto_cli.py     # Main command-line entry point
```

---

## 📝 Input Data Contract

ProtoBackend is highly flexible and supports both **JSONL (line-delimited)** and **Standard JSON (arrays)**.

### Supported Formats:
1.  **Line-delimited JSONL**: Each line is a single JSON object.
2.  **JSON Array**: One large file containing a list of JSON objects.
3.  **Extensions**: Both `.json` and `.jsonl` are automatically detected.

### Minimum Required Fields:
Each record should contain at least these fields:
- `evidence_sentence`: The text snippet to analyze.
- `implicit_aspect`: The target aspect label for training/evaluation.
- `example_id` or `id`: A unique identifier for the record.

**Location**: Place your files in `ProtoBackend/input/<family>/` named as `train.json`, `val.json`, and `test.json`.

---

## 🛠 Advanced Usage

### Hyperparameter Sweep
Find the best configuration for your dataset:
```powershell
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py sweep --dataset-family episodic --thresholds 0.4,0.5,0.6 --topks 1,3,5
```

### Manual Specific Evaluation
Evaluate an existing model on the test split with specific settings:
```powershell
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py eval --dataset-family episodic --threshold 0.55 --top-k 1
```

### Interactive Prediction
Test the system manually:
```powershell
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py predict --sentence "The steak was tender but the service was slow."
```

---

## 🧹 Maintenance

To clean up generated outputs and prune temporary files:
```powershell
backend/venv/Scripts/python.exe ProtoBackend/clean_outputs.py
```
