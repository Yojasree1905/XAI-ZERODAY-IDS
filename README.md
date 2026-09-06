# XAI-ZERODAY-IDS: Explainable AI-Powered Zero-Day Intrusion Detection System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![XAI Engine](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-orange.svg)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [System Architecture & Working Flow](#system-architecture--working-flow)
  - [Architectural Flow Diagram](#architectural-flow-diagram)
  - [Detailed Layer Breakdown](#detailed-layer-breakdown)
- [Core Modules & Operational Features](#core-modules--operational-features)
- [Empirical Performance & Evaluation Benchmarks](#empirical-performance--evaluation-benchmarks)
- [UNSW-NB15 Feature Reference Schema](#unsw-nb15-feature-reference-schema)
- [Repository Structure](#repository-structure)
- [Step-by-Step Installation & Execution Guide](#step-by-step-installation--execution-guide)
- [License](#license)

---

## Executive Summary

**XAI-ZERODAY-IDS** is an enterprise-grade, real-time Intrusion Detection System (IDS) engineered for Next-Generation Security Operations Centers (SOC). Traditional intrusion detection frameworks rely strictly on static signature databases, rendering them ineffective against previously unseen, zero-day cyber threats. Conversely, deep learning and black-box machine learning approaches produce alerts without human-understandable rationale, leading to alert fatigue and delayed incident response.

**XAI-ZERODAY-IDS** addresses both challenges by uniting supervised machine learning classifiers (**XGBoost**, **Random Forest**) with an unsupervised anomaly detector (**Isolation Forest**) within a multi-model consensus architecture. Furthermore, the system integrates **Explainable AI (XAI)** frameworks (**SHAP** and **LIME**), enabling security analysts to inspect global feature importance rankings and local instance-level decision drivers for every detected packet.

---

## System Architecture & Working Flow

### Architectural Flow Diagram

The following diagram illustrates the complete end-to-end data processing, feature transformation, predictive inference, consensus aggregation, and XAI explanation pipeline:

```
+-----------------------------------------------------------------------------------+
|                            STAGE 1: DATA INGESTION                                |
|  +--------------------------+  +-------------------------+  +------------------+  |
|  | Live Traffic Generator   |  | Single Packet Form      |  | Batch Log CSV    |  |
|  | (Dynamic Scenario Engine)|  | (Manual Field Inspector)|  | (PCAP Ingestion) |  |
|  +------------+-------------+  +------------+------------+  +--------+---------+  |
+---------------+-----------------------------+------------------------+------------+
                |                             |                        |
                +-----------------------------+------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                        STAGE 2: PREPROCESSING & SCALING                           |
|  +-----------------------------------------------------------------------------+  |
|  | Categorical Encoding (OneHotEncoder: proto, service, state)                 |  |
|  | Numerical Feature Normalization (StandardScaler: 39 attributes)             |  |
|  | Output: 194-Dimension Dense Feature Alignment Matrix                        |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 3: PARALLEL MODEL INFERENCE ENGINE                      |
|                                                                                   |
|    +----------------------------------+    +----------------------------------+   |
|    |      SUPERVISED DETECTORS        |    |      UNSUPERVISED ESTIMATOR      |   |
|    |                                  |    |                                  |   |
|    |  +----------------------------+  |    |  +----------------------------+  |   |
|    |  | XGBoost Classifier         |  |    |  | Isolation Forest           |  |   |
|    |  | (Primary Gradient Boosted) |  |    |  | (Zero-Day Anomaly Detector)|  |   |
|    |  +----------------------------+  |    |  +----------------------------+  |   |
|    |  | Random Forest Classifier   |  |    |                                  |   |
|    |  | (High-Recall Ensemble)     |  |    |                                  |   |
|    |  +----------------------------+  |    |                                  |   |
|    +----------------+-----------------+    +----------------+-----------------+   |
+---------------------|---------------------------------------|---------------------+
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 4: HYBRID CONSENSUS DECISION ENGINE                     |
|  +-----------------------------------------------------------------------------+  |
|  | 3-Way Rule-Based Consensus Matrix:                                          |  |
|  |  - Normal Verdict: XGBoost=Normal AND RF=Normal AND IsoForest=Normal          |  |
|  |  - Attack Verdict: Supervised Estimators Flag Known Signature Pattern       |  |
|  |  - Zero-Day Anomaly Verdict: Supervised=Normal BUT IsoForest=Outlier          |  |
|  | Risk Score Aggregation: Weighted Confidence Scoring (0% to 100%)              |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 5: EXPLAINABLE AI (XAI) ENGINE                          |
|  +-----------------------------------------------------------------------------+  |
|  | SHAP (SHapley Additive exPlanations): TreeExplainer Global Feature Impact   |  |
|  | LIME (Local Interpretable Model-agnostic Explanations): Local Weights        |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 6: ENTERPRISE STREAMLIT DASHBOARD UI                    |
|  +-------------------+ +--------------------+ +---------------+ +---------------+ |
|  | Live Stream       | | Single Packet      | | Batch Log     | | Engine        | |
|  | Monitor           | | Inspector          | | Analyzer      | | Telemetry     | |
|  +-------------------+ +--------------------+ +---------------+ +---------------+ |
|  +------------------------------------------------------------------------------+ |
|  | XAI Threat Audit & Governance Center                                         | |
|  +------------------------------------------------------------------------------+ |
+-----------------------------------------------------------------------------------+
```

---

### Detailed Layer Breakdown

#### 1. Ingestion Layer
* **Live Dynamic Scenario Engine**: Generates real-time network streams using mathematical distribution sampling (uniform, Gaussian, multi-modal) across 42 network flow features.
* **Single Packet Manual Inspector**: Allows SOC analysts to construct custom packet headers and payload statistics for hypothesis testing.
* **Batch PCAP/CSV Ingestion**: Handles bulk log file uploads, parsing raw attributes into tabular structures.

#### 2. Preprocessing & Feature Engineering Layer
* **Categorical Transformer**: Applies `OneHotEncoder` to categorical attributes (`proto`, `service`, `state`), expanding categorical combinations.
* **Numerical Transformer**: Standardizes 39 numerical features (`dur`, `sbytes`, `dbytes`, `sttl`, `rate`, etc.) using `StandardScaler` to remove scale bias across features.
* **Feature Alignment Pipeline**: Constructs a unified 194-dimension sparse/dense matrix matching the exact input feature format required by the trained estimators.

#### 3. Multi-Model Inference Execution Layer
* **XGBoost Classifier**: A gradient boosted decision tree classifier serving as the primary supervised threat detector for catalogued attack vectors.
* **Random Forest Classifier**: An ensemble of 100 decision trees operating as a high-recall secondary classifier.
* **Isolation Forest**: An unsupervised anomaly estimator trained exclusively on normal traffic distributions to measure structural divergence and isolate zero-day anomalies.

#### 4. Hybrid Consensus & Risk Scoring Layer
* **Consensus Logic**: Combines supervised signature matching with unsupervised anomaly detection according to the following decision matrix:
  - **Normal**: Supervised models evaluate Normal AND Isolation Forest evaluates Normal.
  - **Malicious Threat**: Supervised models detect known attack patterns.
  - **Zero-Day Anomaly**: Supervised models evaluate Normal (signature absent), but Isolation Forest flags an outlier anomaly (structural anomaly).
* **Risk Score Aggregation**: Computes a dynamic threat probability score normalized between 0.0% and 100.0%.

#### 5. Explainable AI (XAI) Interpretability Layer
* **SHAP (SHapley Additive exPlanations)**: Calculates Shapley values via `TreeExplainer` to establish global feature importance rankings and summary impact plots.
* **LIME (Local Interpretable Model-agnostic Explanations)**: Generates local linear surrogate models around individual prediction instances, highlighting positive (threat-increasing) and negative (threat-decreasing) feature contribution weights.

#### 6. Presentation & Dashboard Layer
* **Streamlit UI**: Renders an interactive web interface featuring live streaming tables, risk gauges, metric cards, confusion matrices, audit logs, and XAI visualizations.

---

## Core Modules & Operational Features

1. **Live Network Stream & Generator**:
   - Simulates continuous network traffic streams with adjustable packet volume and scenario presets (Mixed Stream, DDoS Volumetric, Zero-Day Exfiltration, Normal Web Traffic).
   - Renders live packet tables, risk breakdown pie charts, and downloadable stream logs.

2. **Single Packet Threat Inspector**:
   - Interactive form for manual parameter input across Connection Headers, Traffic Volume, Rates & TTL, and Flow Counters.
   - Provides an expanded technical glossary defining all 42 network flow attributes.

3. **Batch Network Log Analyzer**:
   - Processes bulk CSV network log uploads through the full inference pipeline.
   - Generates downloadable security audit reports with recommended security action plans (Block IP, Quarantine, Safe Traffic).

4. **Engine Telemetry & Benchmarks**:
   - Displays empirical evaluation metrics (Accuracy, Precision, Recall, F1-Score) measured across 82,332 test samples from the UNSW-NB15 dataset.
   - Renders metric comparison charts and confusion matrices.

5. **XAI Threat Audit Center**:
   - Global SHAP feature importance bar charts and summary plots.
   - Instance-level LIME feature weight inspection for individual network packets.

---

## Empirical Performance & Evaluation Benchmarks

The hybrid engine was evaluated on the benchmark **UNSW-NB15** network intrusion dataset (175,341 training samples, 82,332 testing samples across 9 attack categories including Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms).

| Model / Architecture | Accuracy | Precision | Recall | F1-Score | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **95.36%** | **95.32%** | **98.00%** | **96.64%** | Primary Supervised Threat Detector |
| **Random Forest Classifier** | 87.04% | 81.64% | 98.65% | 89.34% | High-Recall Baseline Classifier |
| **Isolation Forest** | 32.24% | 51.94% | 7.61% | 13.27% | Unsupervised Zero-Day Anomaly Detector |
| **Hybrid Consensus Ensemble** | **95.82%** | **95.20%** | **98.71%** | **96.91%** | Production Decision Engine |

---

## UNSW-NB15 Feature Reference Schema

The system processes 42 network parameters categorized into:

* **Basic Flow Features**: `dur` (Duration), `proto` (Protocol), `service` (Service), `state` (State), `spkts` (Src Packets), `dpkts` (Dst Packets), `sbytes` (Src Bytes), `dbytes` (Dst Bytes).
* **Packet & Rate Statistics**: `rate` (Packet Rate), `sttl` (Src TTL), `dttl` (Dst TTL), `sload` (Src Load), `dload` (Dst Load), `sloss` (Src Packet Loss), `dloss` (Dst Packet Loss), `sinpkt` (Src Inter-packet Time), `dinpkt` (Dst Inter-packet Time), `sjit` (Src Jitter), `djit` (Dst Jitter), `swin` (Src Window), `dwin` (Dst Window), `stcpb` (Src TCP Sequence Num), `dtcpb` (Dst TCP Sequence Num), `tcprtt` (TCP RTT), `synack` (SYN-ACK Delay), `ackdat` (ACK Delay), `smean` (Mean Src Pkt Size), `dmean` (Mean Dst Pkt Size).
* **Content & Host Features**: `trans_depth` (HTTP Depth), `response_body_len` (HTTP Body Len), `ct_srv_src` (Service-Src Count), `ct_state_ttl` (State-TTL Count), `ct_dst_ltm` (Dst LTM Count), `ct_src_dport_ltm` (Src-Dport Count), `ct_dst_sport_ltm` (Dst-Sport Count), `ct_dst_src_ltm` (Dst-Src Count), `is_ftp_login` (FTP Login Flag), `ct_ftp_cmd` (FTP Cmd Count), `ct_flw_http_mthd` (HTTP Method Count), `ct_src_ltm` (Src LTM Count), `ct_srv_dst` (Service-Dst Count), `is_sm_ips_ports` (Same IP/Port Flag).

---

## Repository Structure

```
XAI-ZERODAY-IDS/
├── data/                         # UNSW-NB15 raw datasets
│   ├── UNSW_NB15_testing-set.csv
│   └── UNSW_NB15_training-set.csv
├── images/                       # Model explainability diagrams
├── notebooks/                    # Exploratory Data Analysis & Model Weights
│   ├── models/                   # Serialized model artifacts (.pkl)
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── XAI_Anlaysis.ipynb
├── results/                      # Evaluation telemetry & confusion matrix plots
├── src/                          # Core Python Engine Modules
│   ├── __init__.py
│   ├── config.py                 # Paths & feature definitions
│   ├── explain.py                # SHAP & LIME XAI engines
│   ├── generator.py              # Dynamic synthetic traffic generator
│   ├── hybrid.py                 # Hybrid consensus decision engine
│   ├── loader.py                 # Model & preprocessor artifact loader
│   ├── model.py                  # Model training logic
│   └── preprocessing.py          # Data cleaning & scaling pipeline
├── ui/                           # Streamlit Web Dashboard
│   └── app.py                    # SOC application interface
├── train_models.py               # Model training entry point script
├── sample_network_data.csv       # Sample batch network logs
├── requirements.txt              # Dependency specifications
├── LICENSE                       # MIT License file
└── README.md                     # Project documentation
```

---

## Step-by-Step Installation & Execution Guide

### Prerequisites
* **Python 3.10+** installed.
* **Git** installed.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Yojasree1905/XAI-ZERODAY-IDS.git
cd XAI-ZERODAY-IDS
```

---

### Step 2: Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: (Optional) Retrain Models
To train model weights on the UNSW-NB15 dataset:

```bash
python train_models.py
```

---

### Step 5: Launch the Streamlit Web Application
Run the interactive application dashboard:

```bash
streamlit run ui/app.py
```

After launching, open your browser and navigate to:
`http://localhost:8501`

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
