# Real-Time AKI Detection Microservice

A lightweight Python microservice for detecting **Acute Kidney Injury (AKI)** from a simulated hospital message stream. The service receives HL7v2 messages over MLLP/TCP, updates patient creatinine history, runs online inference, and sends an HTTP pager alert when AKI risk is detected.

> This repository is based on a university software engineering coursework project. All data used by the project is simulated; no real patient data is included.

## Project Overview

Hospitals receive patient admission, discharge, and laboratory-result events from systems such as PAS and LIMS. This project converts a static AKI prediction model into a deployable real-time inference service:

1. receive HL7v2 messages over an MLLP/TCP stream;
2. parse ADT and ORU messages;
3. maintain creatinine history for each patient;
4. extract sequence-level features from creatinine results;
5. run a pre-trained scikit-learn model;
6. send pager alerts through an HTTP POST request.

## My Role

- Implemented the backend inference service in Python.
- Built the HL7v2 + MLLP message ingestion and ACK handling logic.
- Implemented patient state management and dbm-based persistence.
- Integrated pre-trained model artifacts for online AKI prediction.
- Added unit tests for message framing, HL7 parsing, inference, network ACKs, and persistence behavior.

## Tech Stack

- **Language**: Python
- **ML / Data**: scikit-learn, NumPy, pandas, joblib
- **Networking**: TCP sockets, HL7v2, MLLP, HTTP POST
- **Persistence**: Python dbm
- **Deployment**: Docker, Kubernetes-compatible environment variables
- **Testing**: pytest, unittest

## Repository Structure

```text
.
├── Dockerfile
├── main.py
├── requirements.txt
├── src/
│   ├── data.py                 # HL7 parsing and patient state updates
│   ├── inference.py            # Feature extraction and model inference
│   ├── network.py              # MLLP/TCP client, ACK handling and pager HTTP calls
│   ├── storage.py              # dbm persistence layer
│   ├── inference_model.pkl      # Pre-trained model artifact
│   ├── scaler.pkl              # Feature scaler artifact
│   ├── imputer.pkl             # Imputer artifact
│   └── feature_columns.pkl      # Model feature schema
└── tests/
    ├── test_data.py
    ├── test_inference.py
    └── test_network.py
```

## Key Features

### 1. HL7v2 + MLLP message ingestion

The service connects to an MLLP server through TCP and extracts complete HL7 payloads from the byte stream. It handles common stream-processing issues such as packet fragmentation, sticky packets, invalid framing, and connection interruption.

### 2. ACK-based message consumption

After successfully receiving each message, the service returns a minimal HL7 ACK message wrapped in MLLP framing. This allows the upstream simulator to continue sending the next message.

### 3. Patient state management

The data layer parses simplified ADT and ORU messages, extracts MRN, gender, timestamp, and creatinine values, and maintains per-patient creatinine history in memory.

### 4. Online AKI inference

The inference layer converts creatinine history into model features such as last value, baseline value, ratio, delta, max, min, mean, and slope. It then loads a pre-trained scikit-learn model and returns an AKI risk decision.

### 5. Pager alert integration

When AKI risk is detected, the service sends an HTTP POST request to the pager endpoint with the patient MRN and test timestamp.

### 6. Persistence and restart recovery

The service persists creatinine histories using Python `dbm`, allowing key patient state to be restored after process restart.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLLP_ADDRESS` | MLLP server address in `host:port` format | `localhost:8440` |
| `PAGER_ADDRESS` | Pager HTTP service address | `localhost:8441` |
| `PAGER_PAGE_PATH` | Pager API path | `/page` |
| `HISTORY_PATH` | Historical creatinine CSV path | `/data/history.csv` |
| `DBM_PATH` | dbm persistence file path | `patient_history.db` |

## Running Locally

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Run the service:

```bash
export MLLP_ADDRESS=localhost:8440
export PAGER_ADDRESS=localhost:8441
export HISTORY_PATH=/data/history.csv
python main.py
```

The service expects an MLLP-compatible simulator to be running and a historical creatinine file to be available at `HISTORY_PATH`.

## Running with Docker

Build the image:

```bash
docker build -t aki-detection-service .
```

Run the container:

```bash
docker run --rm \
  -e MLLP_ADDRESS=host.docker.internal:8440 \
  -e PAGER_ADDRESS=host.docker.internal:8441 \
  -v $(pwd)/data:/data \
  aki-detection-service
```

For Linux, replace `host.docker.internal` with the correct host or container network address.

## Running Tests

```bash
pip install pytest
pytest -q
```

The test suite covers:

- MLLP frame extraction;
- sticky packet and partial packet handling;
- HL7 ADT/ORU parsing;
- gender and creatinine extraction;
- inference robustness;
- HTTP pager request behavior;
- dbm persistence and recovery behavior.

## Notes

- This is a prototype for a simulated hospital environment.
- No real patient data, hospital credentials, or private infrastructure configuration is included.
- CSV input files, MLLP message dumps, logs, and local dbm files are intentionally ignored by `.gitignore`.
- Model artifacts are included because they are required for local inference.

## Resume Summary

Built a real-time AKI detection backend service using Python, HL7v2/MLLP, scikit-learn, Docker, and dbm persistence. The service consumes simulated hospital PAS/LIMS messages, performs online creatinine-based inference, and triggers HTTP pager alerts while supporting stream framing, ACK handling, connection recovery, persistence, and unit testing.
