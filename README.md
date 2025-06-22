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
