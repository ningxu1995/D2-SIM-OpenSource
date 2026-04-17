# D²-SIM Hardware: Piezoelectric Adapter / Converter

This repository contains the Computer-Aided Design (CAD) models, mechanical drawings, and assembly files for the piezoelectric adapter module used in the D²-SIM system. 

These files are provided to help researchers replicate, modify, or manufacture the optomechanical mounting structures required for the high-precision operations detailed in our accompanying manuscript.

## File Structure & Description

The repository is organized into the following categories:

### 1. Main Assembly
* `Total.SLDASM`: The complete SolidWorks assembly file of the piezoelectric adapter system.

### 2. Core Piezoelectric Converter Components
* `piezoelectric converter.SLDPRT`: The 3D SolidWorks part model of the custom piezoelectric converter.
* `piezoelectric converter.SLDDRW`: The 2D SolidWorks manufacturing drawing file containing exact dimensions and tolerances.
* `piezoelectric converter.pdf`: A PDF export of the mechanical drawing, provided for quick reference for users without CAD software.

### 3. Associated Optomechanical Mounts
Standard optical mounting components and adapters (referencing standard optomechanical parts, Thorlabs equivalent models):
* **CP33 Series (30 mm Cage Plate):**
  * `CP33T_M-Solidworks.sldprt` (3D Part)
  * `CP33_M-AutoCADPDF.pdf` (Drawing)
* **KC1T Series (Kinematic Mount):**
  * `KC1T-P-Solidworks.sldprt` (3D Part)
  * `KC1T-P-AutoCADPDF.pdf` (Drawing)
* **KCB1C Series (Right-Angle Kinematic Mount):**
  * `KCB1C_M-Solidworks.sldprt` (3D Part)
  * `KCB1C_M-AutoCADPDF.pdf` (Drawing)

### 4. Additional Structural Parts
* `mounting bracket.SLDPRT`: Base mounting plate. 

---

## 💻 Software Requirements
* **3D/2D CAD Files:** SolidWorks 2022 or later is recommended to open the `.SLDASM`, `.SLDPRT`, and `.SLDDRW` files without compatibility issues.
* **PDF Drawings:** Any standard PDF viewer can be used to open the `.pdf` files.


## 📄 License
This hardware documentation is open-sourced under the MIT/CC-BY-4.0 License.