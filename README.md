# D2-SIM: A Full-Stack Open-Source Platform for Computational Nanoscopy

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Open Source](https://img.shields.io/badge/Hardware-Open_Source-orange.svg)]()
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)]()
[![MATLAB R2023b](https://img.shields.io/badge/MATLAB-R2023b-red.svg)]()

Welcome to the official repository for **Dual-frame Diffractive Structured Illumination Microscopy (D2-SIM)**. 

This repository serves as a comprehensive, full-stack open-source archive supporting our manuscript:
> **"Standalone  dual-frame diffractive structured illumination enables sustained live-cell nanoscopy"** > Under Review, 2026.

## 🌟 Project Overview
To address the critical trade-offs between spatiotemporal resolution, instrument footprint, and phototoxicity in live-cell nanoscopy, we developed D2-SIM. By synergizing a pure-phase diffractive optical architecture with a Sparsity-Driven Reconstruction Network (SDRN), D2-SIM fundamentally compresses the physical sampling requirement to merely two frames, achieving a 29.0-ms instantaneous reconstruction while miniaturizing the hardware footprint to a standalone module.

To champion the democratization and reproducibility of advanced microscopy, this repository provides **complete transparency across the entire optomechanical and computational pipeline**. From 3D-printable hardware adapters and optical diffraction simulations to the deep learning inference engine and biophysical quantification scripts, all assets are freely accessible.

## 📂 Repository Architecture (Monorepo)
This repository is organized into five specialized modules. Please navigate to the respective subdirectories for specific code, data, and usage instructions:

```text
D2-SIM-OpenSource/
├── 1-piezoelectric_adapter/    # Hardware: 3D CAD models and mechanical blueprints for optomechanical integration.
├── 2-DOE_light_field_code/     # Optics: MATLAB scripts for inverse phase design and interference field simulation.
├── 3-DOE_fabrication_data/     # Fabrication: L-Edit photolithography layouts and SEM validation data.
├── 4-SDRN_code/                # Deep Learning: PyTorch implementation of the reconstruction network (RCAN + Physics Loss).
└── 5-msd_analysis/             # Biophysics: Python scripts for quantitative trajectory tracking and MSD anomalous diffusion analysis.
🚀 Getting Started
Since this is a full-stack repository encompassing different software ecosystems (Python, MATLAB, CAD), each module contains its own dedicated README.md with detailed environment setups, execution commands, and expected outputs.

To run the Deep Learning model: Please navigate to 4-SDRN_code/ and follow the PyTorch setup instructions. We have provided a minimal toy dataset for one-click validation.

To reproduce the Main Figures (e.g., Fig 4b, 5e): Please navigate to 5-msd_analysis/ to execute the quantitative biophysical diffusion scripts.

📊 Data & Model Availability
Due to file size limits, the extensive synthetic training datasets, the multi-gigabyte raw live-cell acquisition sequences, and the optimal SDRN pre-trained weights (SDRN_best_model.pth) are persistently archived on https://cloud.tsinghua.edu.cn/d/f3a7a0ba9b1045d69368/.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details. Commercial entities wishing to utilize the hardware blueprints or algorithmic pipelines should contact the corresponding authors.