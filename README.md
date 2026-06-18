# SynthGuard: An End-to-End Generative Pipeline for Financial Data Synthesis, Auditing, and Privacy Attack Validation

* **Role:** Independent Student Researcher (B.Sc. CS AI Academic Capstone)
* **Dataset Scale:** 20,000 rows × 27 dimensions (Demographics, 10 Core Spending Vectors, Savings, Disposable Income)
* **Core Stack:** Python, SDV (CTGAN, TVAE, GaussianCopula), Scikit-Learn, XGBoost, LightGBM, Streamlit

SynthGuard is a 7-phase software pipeline built to solve data privacy bottlenecks in financial machine learning. The framework generates production-grade synthetic financial records, executes rigorous statistical quality audits, and simulates adversarial privacy attacks to guarantee zero-leakage compliance before data distribution.

---

### 📊 The 7-Phase Engineering Lifecycle

#### Phase 1: Exploratory Data Analysis (EDA)
Mapped baseline feature distributions and transactional patterns across the 20,000-row real dataset using 7 distinct visualization architectures: Donut, Pie, Funnel, Histograms, Stacked Bar, Grouped Column, and standard Bar charts.

#### Phase 2: Predictive Baseline Modeling
Trained three regression architectures to map dependencies and predict `Desired_Savings`. 
* **Random Forest Regressor:** Achieved peak performance with 97.71% accuracy ($R^2$ = 0.9321)
* **XGBoost Regressor:** 96.80% accuracy
* **LightGBM Regressor:** 96.21% accuracy

#### Phase 3: Generative Data Synthesis (500 Epochs)
Trained three deep generative paradigms via the Synthetic Data Vault (SDV) library to synthesize 20,000 rows each:
* **CTGAN:** Compute time ~73 minutes
* **TVAE:** Compute time ~25 minutes
* **GaussianCopula:** Compute time ~1 minute

#### Phase 4: Multi-Metric Quality Auditing
Ranked generative fidelity using a 5-variable weighted scoring engine (evaluating Kolmogorov-Smirnov Similarity, Mean/Std Deviations, Correlation Matrix Stability, and Random Forest Indistinguishability). **CTGAN emerged as the high-fidelity leader with an overall quality score of 73.03%.**

#### Phase 5: Adversarial Privacy Attack Simulation
Tested vulnerabilities by simulating three production-level adversarial breaches. The pipeline established a **Combined Privacy Protection Score of 82.02%**, proving high resilience against:
* K-Nearest Neighbors (KNN) Proximity Attacks
* Attribute Inference Attacks (Restricted to a baseline random-guess rate of 25.32%)
* Singling Out Attacks

#### Phase 6: Statistical Fidelity Validation (STEAM Rate)
Executed a strict STEAM evaluation framework to check structural data degradation. **CTGAN secured the top validation baseline score of ~80%**, yielding a final integrated composite platform benchmark of **77.56%**.

#### Phase 7: Deployment via the SynthGuard Web App (`app.py`)
Built and deployed an interactive, multi-page **Streamlit application** organized into 7 functional modules:
1. **Secure Login:** User authentication gateway.
2. **Interactive Dashboard:** Live charts tracking real-vs-synthetic distribution shifts.
3. **Upload & Validate:** Drag-and-drop CSV validation engine calculating usability indices.
4. **Train Model:** Dynamic interface to spin up predictive architectures on the fly.
5. **User Input Predictive Engine:** Generates hyper-personalized savings predictions from user metrics.
6. **Contextual AI Chatbot:** Interactive assistant to handle real-time project inquiries.
7. **Audit History Log:** Central ledger to track processing histories and export PDF validation reports.
