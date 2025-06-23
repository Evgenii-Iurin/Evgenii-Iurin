> **Computer Vision | NLP | Recommendation Systems | Data Analysis**

### Hello, I’m a Data Scientist and Machine-Learning Engineer with over 3 years of experience, specializing in Computer Vision, Recommender Systems, Time-Series Forecasting, Data Analysis, and NLP. Below are some projects I have pursued in my spare time alongside my primary role.


### NLP

- **Drone Rescue project** – a multi-agent communication system built on small language models. Each agent is a language model fine-tuned on a custom dataset using the LoRA method. The project is not publicly available, but I can share parts of it upon request.
- **Visual RAG system** – the user asks a question (e.g., “Where can I wash my hands?”) and the system returns an image that matches the description. A VLM, an LLM, and a vector database are employed. The project is complete; a demo is in progress.

### Computer Vision

- [Image Matching](#image-matching)
- Hand Keypoint Detection
- Object Tracking from scratch (Hungarian Algorithm + Kalman Filter)
- YOLOv1 from scratch
- Camera Calibration
- SLAM from scratch
- Differentiable Rasterization
- NeRF (Neural Radiance Fields)

### Recommendation Systems

- Implementing End-to-End Two-Tower model (MLP-based) ; KNN Base Recommendation System ; Cold-Start User Modeling ; A/B Testing

### Data Analysis for business

- Financial Analysis & Pricing ; Churn Prediction

---

## Computer Vision Projects

### [Image Matching](https://github.com/Evgenii-Iurin/ITMO-CV-ADV/tree/dev/src/image_matching)

- **Description:** After downloading a dataset from Roboflow, it was discovered that some training images had leaked into the test set as augmented versions.
- **Task:** Identify and remove "extra" or duplicate images from the test set.
- **Solution:**
  - Used image hashing to quickly eliminate obviously dissimilar images. This significantly reduced computation before the next steps.
  - Extracted descriptors using **SIFT**
  - Matched descriptors using **FLANN** (Fast Library for Approximate Nearest Neighbors)
  - Analyzed the distribution of match thresholds to determine the optimal cutoff for filtering duplicates

### [Hand Keypoint Detection](https://github.com/Evgenii-Iurin/ITMO-CV-ADV/tree/dev/src/keypoint_detection) – [Demo](https://www.linkedin.com/posts/eugene-iurin_%F0%9D%97%98%F0%9D%98%83%F0%9D%97%B2%F0%9D%97%BF%F0%9D%98%86-%F0%9D%97%B1%F0%9D%97%B2%F0%9D%98%83%F0%9D%97%B2%F0%9D%97%B9%F0%9D%97%BC%F0%9D%97%BD%F0%9D%97%B2%F0%9D%97%BF-%F0%9D%97%AE%F0%9D%98%81-%F0%9D%97%B9%F0%9D%97%B2%F0%9D%97%AE%F0%9D%98%80%F0%9D%98%81-activity-7329534223801802752-Q2HA?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD80pnwB9xOyFSB0npQlx6zGxQGo8aoNQv4)

- **Description:** Developed an interactive application for real-time **hand keypoint tracking**. The app enables users to draw using hand gestures.
- **Task:** Implement a hand-tracking model and integrate it into a working drawing application.
- **Solution:**
  - Used **MediaPipe** as a baseline for fast hand detection
  - Detected finger gestures to trigger drawing actions
  - Implemented a custom **U-Net** architecture and trained it on the **FreiHAND** dataset

### [Object Tracking](https://github.com/Evgenii-Iurin/object-tracking-assignment/blob/main/report_en.md)

**Tracker Soft** is a video object tracking system that uses the Hungarian Algorithm and a Kalman Filter. It matches objects across frames, even in the presence of missed detections, and evaluates tracking performance using the **precision** metric.

#### How does it work?

1. **Object Detection:**
    
    For each frame, object center coordinates and bounding boxes are extracted.
    
2. **Object Assignment (Hungarian Algorithm):**
    
    A cost matrix based on IoU is built between current and previous bounding boxes to assign detections to existing tracks.
    
3. **Kalman Filter:**
    
    Predicts the object's next position, smooths noisy detections, and helps maintain consistent tracking even with temporary detection failures.
    
4. **Track Management:**
    - A new track is created if a detection does not match any existing track
    - A track is deleted if it hasn't been updated for more than 10 frames
    - Euclidean distance and IoU are used to evaluate matches
    - Detections without bounding boxes are excluded from tracking

### [YOLO From Scratch](https://github.com/Evgenii-Iurin/ITMO-CV-ADV/blob/dev/src/yolo_from_scratch/report_en.md)

Build a [YOLOv1](https://arxiv.org/pdf/1506.02640) object detector entirely from scratch, mirroring the original paper.

- **Data & Annotations**
    - 182 annotated video frames (121 train / 61 val) created in CVAT
    - Every 10-th video frame selected; standard bounding-box labels
- **Input Format**
    - Images resized to **448 × 448**
    - Each image split into a **7 × 7 grid**
- **Model Output**
    - Tensor shape: **[batch, 7, 7, 14]**
        - 2 boxes × 5 values (x, y, w, h, conf) + 2 class flags per grid cell
- **Loss Strategy**
    - IoU picks the “responsible” box per cell
    - Loss terms: coordinates, size, confidence
    - Signed-sqrt trick prevents NaNs on negative w/h
- **Architecture**
    - 24 convolutional layers with Leaky ReLU — classic YOLO backbone
- **Training Setup**
    - Optimizer: **SGD**, lr = 0.001, momentum = 0.9, weight decay = 5 × 10⁻⁴
    - Hyperparameters match the original YOLO specification

---

## [Recommendation Systems](https://github.com/Evgenii-Iurin/ITMO-RecSys) | [MTS](https://en.wikipedia.org/wiki/MTS_(telecommunications)) | ITMO University

As part of the project, I worked on the development and evaluation of various recommendation system approaches.

#### Key Contributions:

- **End-to-End Two-Tower Model (MLP-based):**
  - Built a dual-tower architecture with separate feature embeddings for users and items
  - Used a customized **triplet loss** with **cosine distance** instead of Euclidean
  - Performed **hyperparameter optimization**
  - Implemented full **data preprocessing pipeline**
  - Developed **MAP@K** as the custom accuracy function
  - Added **model interpretability** via **UMAP visualization** of embeddings

- **KNN-Based Recommendation Model:**
  - Recommended items based on user similarity
  - Designed to handle **cold-start users**

- **Model Benchmarking & Optimization:**
  - Prepared and tuned several baseline models for comparison:
    - Implicit ALS
    - SVD
    - LightFM
  - Hyperparameter tuning for each model

- **Cold-Start User Modeling:**
  - Developed a specialized pipeline for recommendations to new users with sparse interaction history

- **Feature Engineering:**
  - Created and selected features relevant to user-item interaction

- **Custom Evaluation Metrics (from scratch):**
  - Precision@K  
  - Recall@K  
  - MAP@K  
  - AP@K  
  - DCG@K / IDCG@K  
  - NDCG@K  

- **A/B Testing:**
  - Designed and conducted A/B tests to evaluate model performance in production-like scenarios



## [3D Computer Vision Advanced](https://github.com/Evgenii-Iurin/ITMO-3DCV) | School of Data Analysis

Completed the course with a grade of *Satisfactory*. Lectures were taught by [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ&hl=en) (AIRI). The course included the following assignments:

[kmapfmasd](./ITMO-3DCV)

- **Camera Calibration**
  - Camera matrix initialization using the DLT method
  - Refinement via gradient-based optimization of reprojection error
  - Matrix factorization into intrinsic and extrinsic parameters

- **SLAM from Scratch**
  - Full SLAM pipeline implementation to reconstruct a point cloud from a given dataset

- **Differentiable Rasterization**
  - Implemented texture optimization from input images and known camera poses

- **NeRF (Neural Radiance Fields)**
  - Built a basic NeRF model ([paper](https://arxiv.org/pdf/2003.08934.pdf)) without hierarchical sampling

---

## Data Science Course – Karpov Courses

> **Note:** Due to NDA, I cannot share specific code or solutions from the course. Below is an outline of the techniques and topics I worked with.

### Key Areas Covered

####  NLP
- Sentiment analysis on customer reviews using **BERT**

####  Custom Metrics & Loss Functions
- Implemented common metrics: **SMAPE**, **MAPE**, **MAE**
- Designed task-specific losses:
  - **Asymmetric losses** (e.g., RMSLE)
  - **Custom LTV error**: Root Sum Absolute Squared Error
- Classification metrics used in recommender systems:
  - **Recall / Precision / F1-Score / Specificity@K**
- **Triplet Loss** for computer vision tasks

####  Experimentation & Analysis
- A/B testing with **K-Fold** cross-validation (Kaggle-style model evaluation)
- Measured metric lift using **T-test**

####  Forecasting & Modeling
- **Demand forecasting** for cold-start users
- **Dynamic pricing** using competitor data:
  - Applied **metric learning** to match client and competitor products
  - Used product **images and descriptions** as embedding inputs

####  Algorithms Implemented from Scratch
- **PCA**
- **Decision Tree**
- **Gradient Boosting**
- **UCB Algorithm** for auction/arbitrage problems

####  SQL Practice
- WAU (Weekly Active Users) calculation
- DAU (Daily Active Users) query logic

---

## Data Analysis Project: Stretching Studio

**Project Goal:** Analyze user behavior and financial data to improve retention and pricing strategy for a stretching studio.

> **Note:** The project was conducted under NDA due to the use of sensitive financial data. While specific results and datasets are confidential, I can share my overall analytical approach and methodology.

### Key Contributions:
- **Data Cleaning:** Processed and structured raw user and transaction data for further analysis.
- **Churn Prediction:** Developed a model to identify users unlikely to renew their subscription, using behavioral and transactional features.
- **Financial Analysis & Pricing:** Conducted exploratory financial analysis and provided recommendations on pricing optimization based on customer value segmentation.

**Approach Overview:**
- Defined KPIs and target labels (e.g., churned users)
- Performed feature engineering based on user activity and payment patterns
- Evaluated model performance and interpreted results
- Presented insights and pricing suggestions to stakeholders

**Tech stack:** Python (Pandas, NumPy, scikit-learn), SQL, Jupyter Notebook, matplotlib/seaborn


## Statistics

Worked on real-world cases using datasets from an online streaming platform focused on series and films (KION). Due to an NDA, project notebooks cannot be shared publicly, but selected parts can be demonstrated during a call (permission granted).

| Category                      | Topics                                                                 |
|------------------------------|------------------------------------------------------------------------|
| Descriptive Statistics       | Std. deviation, quartiles, 3-sigma rule, correlation                   |
| Confidence Intervals         | Intervals for means, outlier detection, avg. receipt, purchase frequency, returning users |
| Hypothesis Testing           | H₀/H₁, p-value, t-test, Mann–Whitney U test                            |
| Normality Checks             | Q-Q plot, Kolmogorov–Smirnov test, Shapiro–Wilk test                   |
| Product Metrics              | ARPU, user retention, avg. receipt, user activity                      |




<!--
Archive
[![yolov7_training](src/pic/yolov7_training.png)](https://github.com/)
[![waymo_preparation_dataset_to_yolo_format](src/pic/waymo_dataset_preparation_new.png)](https://github.com/)
[![mean_median_filters](src/pic/mean_median_filters.png)](https://github.com/)
[![collecting_class_name](src/pic/class_collecting_new.png)](https://github.com/)
[![dataset_class_distribution](src/pic/dataset_distribution_new.png)](https://github.com/)
-->
