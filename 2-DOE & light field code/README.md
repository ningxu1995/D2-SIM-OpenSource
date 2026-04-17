# D²-SIM Software: DOE Design and Light Field Code

This repository contains the source code for the design, simulation, and optimization of the Diffractive Optical Elements (DOEs) and the associated light field modulation used in the D²-SIM system. 

These scripts are essential for generating the precise phase masks, calculating light field propagation, and executing the phase retrieval algorithms required to achieve high-fidelity super-resolution imaging across multispectral and hyperspectral domains.

## 📁 Repository Structure

The code is organized into the following main modules:

### 1. DOE Phase Mask Generation
Scripts in this directory are used to compute and generate the phase patterns for the diffractive optical elements.
* `generate_phase.m`: Main script to calculate the optimal phase distribution for the required illumination patterns.

### 2. Light Field Simulation & Propagation
Code for simulating the forward propagation of the optical field through the D²-SIM system.
* `sim2d_experiment .m': Simulates the interaction of the illumination beam with the DOE and its subsequent propagation to the sample plane.

### 3. Data Processing & Analysis
* `Get_system_parameter.m`: Scripts for analyzing the simulated or experimentally captured light field intensity distributions, including modulation contrast evaluation. Configuration files defining crucial optical parameters (wavelengths, pixel pitch, focal lengths, and system magnification).

## 💻 System Requirements & Dependencies

* **Programming Language:** MATLAB R2022b or newer
* **Required Libraries/Toolboxes:** * Matplotlib
    *  Image Processing Toolbox for MATLAB

## 🚀 Quick Start
1.  Run `Get_phase` to generate a sample DOE phase mask and visualize the simulated light field at the focal plane. 

## 📄 License
This codebase is distributed under the MIT/CC-BY-4.0 License.