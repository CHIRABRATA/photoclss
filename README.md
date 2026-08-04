# 📸 Trip Photo Organizer AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/InsightFace-ArcFace-blueviolet?style=for-the-badge" alt="InsightFace">
  <img src="https://img.shields.io/badge/scikit--learn-DBSCAN-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

<p align="center">
An automated, AI-powered Streamlit web application that scans a batch of unorganized trip photos, detects human faces, extracts 512-dimensional feature embeddings, clusters individuals using DBSCAN, and organizes the photos into person-wise folders and group photo directories.
</p>

---

## 🧭 Pipeline Overview

```mermaid
flowchart LR
    A["📤 Upload<br/>Trip Photos"] --> B["🧠 Face Detection<br/>InsightFace buffalo_s"]
    B --> C["🔢 Extract 512-D<br/>ArcFace Embeddings"]
    C --> D["🧩 DBSCAN Clustering<br/>Cosine Distance"]
    D --> E["📁 Build Folders<br/>Person_1, Person_2, ..."]
    D --> F["👥 Group_Photos<br/>Multi-face Shots"]
    E --> G["📦 Zip & Download"]
    F --> G

    style A fill:#1a1a2e,stroke:#00f5ff,color:#fff
    style B fill:#1a1a2e,stroke:#00f5ff,color:#fff
    style C fill:#1a1a2e,stroke:#00f5ff,color:#fff
    style D fill:#16213e,stroke:#ff00ff,color:#fff
    style E fill:#1a1a2e,stroke:#00f5ff,color:#fff
    style F fill:#1a1a2e,stroke:#00f5ff,color:#fff
    style G fill:#0f3460,stroke:#00f5ff,color:#fff
```

---

## 🚀 Key Features
* **Face Detection & Vector Extraction:** Utilizes InsightFace (`buffalo_s` ONNX model) to extract normalized 512-D ArcFace face embeddings.
* **Unsupervised Clustering:** Uses DBSCAN with cosine distance metric to group identical individuals without requiring pre-labeled training data.
* **Smart Folder Structuring:** Automatically generates person-specific directories (`Person_1`, `Person_2`, etc.) and a dedicated `Group_Photos` folder for multi-face shots.
* **One-Click Download:** Compresses the full organized folder structure into a downloadable `.zip` file directly within Streamlit.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Frontend/UI:** Streamlit
* **Computer Vision & AI:** InsightFace, OpenCV, NumPy, Pillow
* **Clustering Engine:** scikit-learn (DBSCAN)

---

## 📂 Project Structure
```text
project/
├── app.py                  # Main Streamlit UI & pipeline controller
├── requirements.txt        # Project dependencies
├── utils/
│   ├── detector.py         # InsightFace model loading & embedding extraction
│   ├── cluster.py          # DBSCAN face embedding clustering
│   ├── organizer.py        # Disk directory builder & image file copying
│   └── zipper.py           # In-memory ZIP archive generation
└── README.md
```