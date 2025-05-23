# LENS: Lightweight and Explainable LLM-Based APT Detection at the Edge for 6G Security

This repository contains the source code, experimental setup, and modified baselines for the paper:

📄 **"LENS: Lightweight and Explainable LLM-Based APT Detection at the Edge for 6G Security"**

---

## 📁 Repository Structure

- `subset5.py`, `subset30.py`, `subset100.py`: Run experiments on different portions of the dataset (5%, 30%, and 100%).
- `subset30_original.py`: Final implementation used to generate the main results in the paper.
- `deeplog_modified/`, `earlycrow_modified/`: Modified versions of the baseline models DeepLog and EarlyCrow.
- `notes/`: Contains original experiment notes (initially written in Turkish, now translated).

---

## 🧪 Experiment Summary

- Experiments were conducted on 3 dataset subsets: 5%, 30%, and 100%.
- Final paper results are based on `subset30_original.py`.
- Focus: APT detection in IIoT environments, targeted for edge-based 6G cybersecurity architectures.

---

## 🌐 Dataset: CICAPT-IIOT 2024

The experiments are based on the **CICAPT-IIOT 2024** dataset, which is designed to simulate advanced persistent threat (APT) scenarios in Industrial IoT systems.

- 📄 Citation:
  > Ghiasvand, Erfan, et al.  
  > *"CICAPT-IIOT: A provenance-based APT attack dataset for IIoT environment."*  
  > arXiv preprint [arXiv:2407.11278](https://arxiv.org/abs/2407.11278) (2024)

- 🔗 The dataset is publicly available online. (Search: "CICAPT-IIOT 2024")

---

## ⚙️ Edge & Cloud Deployment

- **Edge Platform**:  
  Evaluated on a **Raspberry Pi**, demonstrating the feasibility of lightweight APT detection on edge devices.

- **LLM Runtime with Lightning.ai**:  
  Due to long execution times for LLM components, we utilized [Lightning.ai](https://lightning.ai/), which offers **15 hours of free cloud compute**—ideal for prototyping and experimentation.

---

## 🔧 Baseline Enhancements

Modified versions of:
- **DeepLog**
- **EarlyCrow (a.k.a. EarlyBird)**

Enhancements include:
- Performance tuning
- Data preprocessing improvements
- Compatibility fixes for our use case

Each version is included in its own folder with related scripts and documentation.

---

## 🚀 How to Run

Install the required dependencies (a `requirements.txt` may be included later), then run:

```bash
# Run the final experiment used in the paper
python subset30_original.py
