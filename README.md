# Land Type Classification using Sentinel-2 Satellite Images

## Project Overview

This repository contains the code and documentation for the Land Type Classification using Sentinel-2 Satellite Images project, led by Eng. Ahmed M. Fayad and Eng. Mohammad Mostafa. The project focuses on leveraging Deep Neural Networks (DNNs) to classify various land types—such as agriculture, water, urban areas, desert, roads, and trees—using multispectral satellite imagery from the European Space Agency's Sentinel-2 mission.

Sentinel-2 provides freely available, high-resolution images ideal for land use classification. The resulting model will support applications in urban planning, environmental monitoring, and resource management.

The project utilizes open-source datasets, such as the EuroSat Dataset, and may involve generating custom labeled datasets using tools like QGIS.

## Objectives

The primary objective is to develop a DNN model that accurately classifies land types based on Sentinel-2 imagery. Key applications include:

- Urban planning
- Environmental monitoring
- Resource management

## Milestones

The project is structured into five milestones, each with specific tasks and deliverables.

### Milestone 1: Data Collection, Exploration, and Preprocessing

**Objectives:** Collect and preprocess satellite imagery data for land classification tasks.

**Tasks:**
- **Data Collection:**
  - Download Sentinel-2 images for a target region (e.g., Egypt) from public repositories like Copernicus Open Access Hub or USGS Earth Explorer
  - Optionally use the EuroSat Dataset for labeled satellite images
  - Ensure multispectral bands (Red, Green, Blue, Near Infrared, etc.) are included

- **Data Exploration:**
  - Conduct exploratory data analysis (EDA) to assess image composition and spectral band relevance
  - Identify issues like imbalanced classes or missing data
  - Visualize sample images and their spectral signatures

- **Preprocessing and Feature Engineering:**
  - Resize images, adjust spectral bands, and normalize data
  - Use QGIS to create additional labeled data if needed
  - Split data into training, validation, and test sets
  - Calculate features like NDVI for enhanced classification
  - Apply image augmentation (e.g., rotations, flips) for dataset diversity

- **Exploratory Data Analysis (EDA):**
  - Generate histograms, scatter plots, and heatmaps to explore spectral band patterns

**Deliverables:**
- EDA Report
- Cleaned Dataset
- Visualizations

### Milestone 2: Advanced Data Analysis and Model Selection

**Objectives:** Perform in-depth analysis and select suitable classification models.

**Tasks:**
- **Advanced Data Analysis:**
  - Analyze spectral band correlations with land types
  - Investigate seasonal or temporal trends if multi-temporal data is available
  - Apply dimensionality reduction (e.g., PCA) to optimize features

- **Model Selection:**
  - Choose DNN models, starting with a simple CNN and exploring architectures like ResNet or VGG
  - Experiment with transfer learning using pre-trained models (e.g., ImageNet, EuroSat)

- **Data Visualization:**
  - Visualize band-land type correlations, confusion matrices, and precision-recall curves

**Deliverables:**
- Data Analysis Report
- Model Selection Summary
- Data Visualizations

### Milestone 3: Model Development and Training

**Objectives:** Build, train, and optimize a DNN model for land type classification.

**Tasks:**
- **Model Development:**
  - Implement a CNN or DNN using TensorFlow or PyTorch
  - Start with a simple architecture and add complexity as needed (e.g., dropout, augmentation)

- **Model Training:**
  - Train the model on the prepared dataset
  - Use early stopping and cross-validation to prevent overfitting
  - Test various batch sizes, learning rates, and optimizers

- **Model Evaluation:**
  - Assess performance with metrics like accuracy, precision, recall, F1-score, and confusion matrix
  - Use class activation maps (CAM) to interpret model focus areas

- **Hyperparameter Tuning:**
  - Optimize using Grid Search or Random Search

**Deliverables:**
- Model Code
- Training and Evaluation Reports
- Final Model

### Milestone 4: Deployment and Monitoring

**Objectives:** Deploy the model for practical use and monitor its performance.

**Tasks:**
- **Model Deployment:**
  - Deploy as a web service or API using Flask, FastAPI, or Django
  - Optionally host on cloud platforms (e.g., AWS, Azure, Google Cloud)

- **Monitoring Setup:**
  - Track performance and detect model drift with monitoring tools
  - Set up alerts for performance drops

- **Model Retraining Strategy:**
  - Plan periodic retraining with new data or user feedback

**Deliverables:**
- Deployed Model
- Monitoring Setup
- MLOps Report

### Milestone 5: Final Documentation and Presentation

**Objectives:** Summarize the project and present it to stakeholders.

**Tasks:**
- **Final Report:**
  - Document the entire process, results, and applications

- **Final Presentation:**
  - Create an engaging presentation with a live demo of the deployed model

- **Future Improvements:**
  - Suggest enhancements like adding Landsat data or using Transformers

**Deliverables:**
- Final Project Report
- Final Presentation

## Tools and Technologies

- **Data Preprocessing:** QGIS
- **Model Development:** TensorFlow or PyTorch
- **Deployment:** Flask, FastAPI, or Django
- **Cloud Platforms:** AWS, Azure, or Google Cloud (optional)

## Dataset

The dataset comprises Sentinel-2 multispectral images from public repositories (e.g., Copernicus Open Access Hub, USGS Earth Explorer) or open datasets like EuroSat. It includes spectral bands such as Red, Green, Blue, and Near Infrared. Custom labeled data may be created using QGIS.

## Team Members

- Eng. Ahmed M. Fayad
- Eng. Mohammad Mostafa

## Project Structure

```
├── data/                # Dataset and preprocessed data
├── notebooks/           # Jupyter notebooks for EDA and model development
├── src/                 # Source code for training, evaluation, and deployment
├── docs/                # Documentation, reports, and presentations
└── README.md            # This file
```

## How to Use This Repository

1. Clone the repository:
```bash
git clone https://github.com/your-repo-url.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the code:
   - Follow instructions in the notebooks/ or src/ folders for preprocessing, training, or deployment tasks.

## Conclusion

This project delivers a robust DNN-based solution for classifying land types using Sentinel-2 imagery. Through structured milestones, it ensures a comprehensive approach from data collection to deployment, offering valuable insights for urban planning, agriculture, and environmental conservation.
