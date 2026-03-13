# MANZAR: Multi-Sensor Deep Learning Flood Detection

This repository contains the Flood Detection Module for the **MANZAR Geospatial Platform**. It implements a highly scalable, multi-sensor deep learning pipeline designed to perform all-weather binary semantic segmentation of floodwaters.

By fusing Synthetic Aperture Radar (SAR) and multispectral optical data, this system overcomes the primary limitation of traditional satellite monitoring: cloud cover during severe monsoon and storm events.

## ⚙️ Architecture & Methodology

The model operates on a unified spatial grid, taking perfectly co-registered 10m-resolution imagery and predicting a binary flood mask (Flooded vs. Non-Flooded) within a change-detection framework.

### Sensor Fusion Stack

The input tensor is strictly locked to a 7-channel sequence to ensure architectural consistency:

1. **SAR Co-Polarization** (Sentinel-1 VV or HH)
2. **SAR Cross-Polarization** (Sentinel-1 VH or HV)
3. **Optical Blue** (Sentinel-2 B02)
4. **Optical Green** (Sentinel-2 B03)
5. **Optical Red** (Sentinel-2 B04)
6. **Optical Near-Infrared** (Sentinel-2 B08)
7. **Scene Classification Layer** (Sentinel-2 SCL Mask)

### Deep Learning Models

* **Baseline Architecture:** A standard U-Net with a ResNet-34 encoder pre-trained on ImageNet. Skip connections preserve the high-resolution spatial data necessary for delineating narrow river channels and urban inundation.
* **Advanced Architecture:** A Dual-Stream Attention U-Net. This features two parallel encoders (one for SAR, one for Optical) to extract sensor-specific features before fusing them at the bottleneck. Attention gating suppresses irrelevant noise (like SAR speckle), while a Feature Pyramid Network (FPN) decoder integrates multi-scale features.

### Strategic Training Optimization

Flood detection suffers from severe class imbalance. This pipeline mitigates this using:

* **Focal Tversky Loss (FTL):** Replaces standard cross-entropy. By tuning the $\alpha$ and $\beta$ parameters (e.g., $\beta=0.7$, $\alpha=0.3$), the network heavily penalizes false negatives, prioritizing recall to ensure critical flood zones are not missed.
* **Multi-Sensor Modality Drop-out:** During training, the optical stream is randomly dropped out (set to zero) with a 20% probability. This forces the network to rely on the SAR backscatter, ensuring the model remains robust during heavy cloud cover.

## 📊 Dataset Acquisition

The model is trained on a combination of global benchmarks and highly localized, high-severity disaster data.

1. **Sen1Floods11 (Global Generalization):** An automated pipeline is included to sync the 4,800+ hand-labeled 512x512 chips across 14 biomes directly from Google Cloud Storage.
2. **2022 Pakistan Mega-Floods (Regional Fine-Tuning):** Using the Microsoft Planetary Computer STAC API, the system automatically pulls, calibrates, and stacks time-series data over disaster zones aligned with authoritative UNOSAT and CEMS vector masks.

### Automated Physical Preprocessing

Raw satellite data is scientifically unusable for deep learning. The ingestion pipeline (`fetch_fusion_ready_imagery`) enforces strict radiometric calibration on the fly:

* **SAR Calibration:** Converts linear power to Decibels ($dB = 10 \times \log_{10}(DN)$), and applies Min-Max scaling from `[-30, 0] dB` to a `[0.0, 1.0]` tensor space.
* **Optical Scaling:** Level-2A surface reflectance is normalized by dividing by 10,000, clipped to `[0.0, 1.0]`. NoData values (`-32768`) are stripped to prevent gradient explosions.
* **Categorical Preservation:** The SCL mask uses nearest-neighbor temporal compositing to preserve discrete integer classes (0-11) rather than corrupting the labels with spatial averaging.

## 🚀 Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/yourusername/MANZAR-Flood-Detection.git
cd MANZAR-Flood-Detection

# Install core geospatial and deep learning dependencies
pip install -r requirements.txt

```

*Required packages include: `torch`, `segmentation-models-pytorch`, `pystac-client`, `planetary-computer`, `odc-stac`, `rioxarray`, `xarray`, `rasterio`.*

## 💻 Usage

### 1. Download the Sen1Floods11 Training Corpus

Run the automated GCP extraction script to pull the hand-labeled baseline data.

```bash
python scripts/download_sen1floods.py

```

### 2. Execute Bulk Multi-Sensor Extraction

To fetch localized, co-registered SAR and Optical stacks for custom events (e.g., 2022 Pakistan Floods), run the STAC extraction manager. This includes built-in rate-limit handling and exponential backoff.

```bash
python scripts/bulk_stac_downloader.py

```

### 3. Run Pre-Flight Quality Assurance

Geospatial data is inherently noisy. Before passing tensors to the PyTorch DataLoader, run the QA checker to flag corrupted pixels, mismatched resolutions, or high-NaN ratios.

```bash
python scripts/dl_preflight_qa.py --dir ./data/raw_flood_stacks

```

### 4. Train the Model

```bash
python train.py --config configs/unet_resnet34.yaml

```

## 📈 Evaluation Metrics

The model is evaluated using standardized computer vision metrics for remote sensing:

* **Intersection over Union (IoU):** Target > 0.75 for regional applicability.
* **Critical Success Index (CSI):** Used specifically for benchmarking against Copernicus GFM algorithms.
* **Precision & Recall:** Tracked independently to balance false alarms against missed disaster zones.

## 📝 License

This project is developed as part of the MANZAR Geospatial Intelligence Platform. Sentinel data is provided freely by the European Space Agency (ESA) via the Copernicus Programme.
