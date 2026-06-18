**SynditAudit: Automated Quality Assurance and Statistical Alignment Framework for Synthetic Datasets**

* **Role:** Lead Developer & Architect
* **Tech Stack:** Python, Pandas, NumPy, SciPy, Scikit-Learn, SDV, Streamlit

### Project Overview
Architected an end-to-end Python evaluation framework to audit the fidelity, privacy, and mathematical utility of generative synthetic datasets.

### Core Features
* **Statistical Alignment:** Implemented non-parametric statistical metrics, including the Kolmogorov-Smirnov (KS) test and Chi-Square test, to evaluate feature distribution alignment between real and synthetic baselines.
* **Correlation Fidelity:** Developed correlation matrix mapping scripts to verify that multi-variable relationships and dependencies remain statistically stable across generated data tables.
* **Privacy Auditing:** Integrated privacy-leak protection algorithms to systematically identify and flag duplicated record leakage from training sets, ensuring zero-compromise privacy compliance.
* **Interactive Reporter:** Built an interactive Streamlit dashboard that automatically processes dual datasets and generates a downloadable PDF Quality Audit Report detailing a custom Data Utility Score.
