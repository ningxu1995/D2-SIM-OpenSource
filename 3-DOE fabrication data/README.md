# D²-SIM Fabrication Files: DOE GDSII Layouts and Phase Data

This repository contains the physical fabrication layouts (GDSII format), image previews, and raw MATLAB data for the Diffractive Optical Elements (DOEs) used in the D²-SIM system. 

These files provide the exact phase modulation patterns required for manufacturing the diffractive optics to achieve illumination lattice.

## 📁 File Structure & Description

The files are categorized by their file types and intended use in the nanofabrication process:

### 1. Raw Phase Data (MATLAB)
* `phi_1000.mat`: This data file contains the computed and optimized phase distribution array. It represents the theoretical continuous or multi-level phase profile before quantization for fabrication.

### 2. GDSII Fabrication Layouts
These are the industry-standard layout files used directly for Electron-Beam Lithography (EBL) or photolithography mask generation. 
* `phi_mask_1.gds`: GDSII layout for Mask Layer 1. (for incident wavelength of 405 nm, 255 phase steps, 8-level lithography steps)
* `phi_mask_2.gds`: GDSII layout for Mask Layer 2. (for incident wavelength of 488 nm, 255 phase steps, 8-level lithography steps)
* `phi_mask_3.gds`: GDSII layout for Mask Layer 3. (for incident wavelength of 561 nm, 255 phase steps, 8-level lithography steps)

### 3. Mask Previews (BMP)
Binarized or quantized preview images corresponding to the GDSII layouts. These are useful for quick visual inspection of the pattern structures without needing specialized layout viewing software.
* `phi_mask_1.bmp`
* `phi_mask_2.bmp`
* `phi_mask_3.bmp`

---

## 💻 Software & Fabrication Notes

* **Viewing GDSII Files:** Open-source layout viewers commercial software L-Edit are recommended for opening and inspecting the `.gds` files.
* **Accessing MAT Files:** `phi_1000.mat` can be loaded into MATLAB using the standard `load()` function, or in Python using `scipy.io.loadmat()`.
* **Fabrication Parameters:** For specific details regarding the substrate material, etching depth, and minimum feature size (CD) required to manufacture these DOEs, please refer to the Supplementary materials  section of our manuscript.

## 📄 License
These fabrication files and data are open-sourced under the CC-BY-4.0 License.