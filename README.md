# Mini Projects of Machine Learning and Deep Learning

A collection of academic and experimental projects exploring machine learning, deep learning, computer vision, natural language processing, sensor-based classification, and industrial defect detection.

**Repository:** [MINI-Projects-of-ML](https://github.com/sparshpatidar2005/MINI-Projects-of-ML)

## About This Repository

This repository brings together multiple machine learning and artificial intelligence projects developed through practical experimentation with datasets, model architectures, preprocessing pipelines, training strategies, evaluation metrics, and visualization techniques.

The collection includes a breast cancer detection study, an ensemble-based sensor classification notebook, a classical computer vision system for traffic violation detection, legal NLP for judgment prediction and clause analysis, and an autoencoder-assisted wafer defect classification framework.

Each notebook represents a separate project or experiment. The breast cancer notebooks and report document different approaches to the same underlying classification problem, allowing the architectures and evaluation methods to be studied individually.

## Projects at a Glance

| Project | Main Techniques | Objective |
|---|---|---|
| Breast Cancer Detection — CNN | Custom CNN | Classify histopathological images as benign or malignant |
| Breast Cancer Detection — Multi-Head Attention | CNN + self-attention | Explore attention-enhanced image classification |
| Breast Cancer Detection — Pretrained Model | ResNet50 transfer learning | Fine-tune a pretrained network for breast cancer classification |
| Sensor-Based Classification Ensemble | IMU, thermal and ToF sensor models | Combine multiple prediction approaches for sequence classification |
| Traffic Red-Light Violation Detection | Classical computer vision and OCR | Detect stop-line violations from traffic video |
| Legal NLP for Judgment Prediction | TF-IDF, ML, LSTM, GRU, BERT and DistilBERT | Explore legal text classification |
| Wafer Map Pattern Intelligence | Convolutional autoencoder and CNN | Classify wafer maps into defect-pattern categories |

---

# 1. Breast Cancer Detection Using a Custom CNN

**Notebook:** `CNN_DA1_AI.ipynb`

**Supporting report:** `AI_Project_FINAL.docx`

## Objective

This project investigates the classification of histopathological breast tissue images into two categories: **benign** and **malignant**. It uses a custom convolutional neural network to learn visual patterns from microscopic images.

The experiments consider images captured at four magnification levels: 40X, 100X, 200X and 400X.

## Dataset and preprocessing

The notebook uses labelled PNG images organized by class and magnification. The image paths, class labels and magnification levels are collected into a Pandas DataFrame.

The preprocessing pipeline includes:

- Resizing images to 224 × 224 pixels.
- Applying ResNet50-compatible image preprocessing.
- Using a stratified training, validation and testing split of 70%, 15% and 15%.
- Applying horizontal and vertical flips, brightness changes and contrast adjustments to training images.
- Creating efficient TensorFlow data pipelines with batching and prefetching.

The recorded split contains 5,535 training images, 1,187 validation images and 1,187 test images.

## Model architecture

The custom CNN consists of convolutional blocks with increasing channel sizes, batch normalization, max pooling and dropout. Global average pooling and dense layers are used for the final binary classification.

The model is trained using the Adam optimizer and binary cross-entropy loss. Early stopping, model checkpointing and learning-rate reduction help control training and preserve promising weights.

## Evaluation

The notebook evaluates the model using:

- Accuracy, precision, recall and F1-score.
- ROC-AUC and precision-recall AUC.
- Confusion matrices and classification reports.
- Training and validation curves.
- Separate performance analysis for each magnification level.

### Recorded results

| Metric | Result |
|---|---:|
| Test accuracy | 85% |
| Test ROC-AUC | 0.9518 |
| Test PR-AUC | 0.9786 |
| Best validation PR-AUC | 0.9761 |

The recorded test recall for the malignant class is approximately 0.97, while benign-class recall is approximately 0.60. This difference is important when evaluating a medical image classifier: overall accuracy alone does not describe how reliably each class is identified.

## Technologies

Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib and Seaborn.

---

# 2. Breast Cancer Detection with Multi-Head Self-Attention

**Notebook:** `Multi_Head_Attention_DA2.ipynb`

## Objective

This experiment extends the breast cancer image-classification work by investigating whether multi-head self-attention can help a CNN represent relevant relationships in histopathological images.

Instead of relying only on convolutional feature extraction, the model introduces an attention mechanism to learn interactions between image features.

## Methodology

The notebook uses the same general image-classification setting as the custom CNN experiment:

- Binary benign-versus-malignant classification.
- Images resized to 224 × 224 pixels.
- Four magnification levels: 40X, 100X, 200X and 400X.
- Training, validation and testing partitions.
- Image augmentation and preprocessing.
- Evaluation through classification reports, confusion matrices, ROC curves and precision-recall curves.

The notebook implements a custom multi-head self-attention layer and experiments with different numbers of attention heads, including two-head and four-head configurations.

## Why multi-head attention?

Different attention heads can learn different relationships among the input features. In image classification, this provides a way to investigate whether combining convolutional features with attention improves the representation used for the final decision.

The experiment is useful for comparing a more conventional CNN with an attention-enhanced architecture.

## Recorded evaluation

The notebook includes a test-set classification report for the two-head configuration:

| Metric | Recorded result |
|---|---:|
| Test accuracy | 89% |
| Benign-class recall | 0.86 |
| Malignant-class recall | 0.91 |
| Weighted F1-score | 0.89 |

These figures refer to the recorded two-head evaluation, not necessarily the four-head model. The notebook should be rerun and the corresponding outputs checked before claiming a final best configuration.

## Technologies

Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib and Seaborn.

---

# 3. Breast Cancer Detection Using Pretrained ResNet50

**Notebook:** `Pre_train_DA1.ipynb`

## Objective

This experiment applies transfer learning to histopathological breast cancer classification using ResNet50, a convolutional neural network pretrained on a large image dataset.

The objective is to reuse learned visual representations and fine-tune the model for the target task of distinguishing benign from malignant tissue images.

## Methodology

The notebook implements the following workflow:

1. Collect image paths, labels and magnification metadata.
2. Resize images to 224 × 224 pixels and apply ResNet50 preprocessing.
3. Create stratified training, validation and test subsets.
4. Load pretrained ResNet50 weights.
5. Add a task-specific classification head and configure trainable layers.
6. Train with Adam and binary cross-entropy.
7. Use early stopping, model checkpointing and learning-rate scheduling.
8. Evaluate the model on the test set and separately by magnification.

The experiment uses accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC, together with confusion matrices and learning curves.

## Recorded results

The notebook reports the following test-set performance:

| Metric | Result |
|---|---:|
| Accuracy | 98.65% |
| Precision | 98.66% |
| Recall | 98.65% |
| F1-score | 98.65% |
| ROC-AUC | 0.9984 |
| Best validation PR-AUC | 0.9993 |

The notebook's test classification report also shows high precision and recall for both benign and malignant classes.

These are the results recorded in this experiment. They should not be interpreted as evidence of clinical readiness or guaranteed performance on independent hospitals, scanners or patient populations.

## Why transfer learning?

Training a deep CNN from scratch can require substantial data and computation. Transfer learning starts from learned image representations and adapts them to a new classification task. Fine-tuning can improve task-specific features, although performance depends on the dataset, split design and training procedure.

## Technologies

Python, TensorFlow, Keras, ResNet50, NumPy, Pandas, Scikit-learn, Matplotlib and Seaborn.

---

# 4. Sensor-Based Classification Using an Ensemble of Three Solutions

**Notebook:** `cmi-ensemble-of-3-solutions.ipynb`

## Objective

This notebook explores the combination of multiple modelling approaches for classifying sequential sensor data. It uses inertial measurement unit (IMU), thermal and time-of-flight (ToF) sensor information.

The notebook references sensor-classification approaches involving feature engineering, neural sequence models and a gated GRU-based solution.

## Data and methodology

The notebook includes three model solutions and a combined prediction function. The overall workflow includes sensor-data processing, model-specific inference, prediction comparison and an ensemble-style combination of predictions.

The referenced approaches include:

- Sensor feature engineering and model blending.
- Neural modelling of multivariate sensor sequences.
- Gated recurrent units (GRUs) for sequential information.
- Sensor modalities including IMU, thermal and ToF measurements.

The notebook references external Kaggle resources for the underlying competition and modelling approaches.

## Why use an ensemble?

Different models may learn different characteristics of a sequence. Combining their predictions can potentially improve robustness when the individual models make complementary errors. The effectiveness of an ensemble must be established through validation and test evaluation rather than assumed.

## Evaluation

The notebook contains separate prediction implementations and a final combination of their outputs. Its final performance should be reported using the relevant competition metric or held-out evaluation after the notebook has been run in the intended environment.

**Note:** The notebook's title and references establish the ensemble approach, but a single verified final score is not specified here.

## Technologies

Python, NumPy, Pandas, Scikit-learn, PyTorch and sensor-processing libraries used by the individual model implementations.

---

# 5. Traffic Red-Light Violation Detection Using Classical Computer Vision

**Notebook:** `Machine_Vision_lab_final.ipynb`

## Objective

This project develops a computer vision pipeline for identifying vehicles that cross a designated stop line while the traffic signal is red.

Unlike deep-learning-based object detection projects, this notebook focuses on classical image-processing and computer-vision techniques.

## Processing pipeline

The notebook describes an end-to-end workflow involving:

1. **Red-light detection:** HSV colour thresholding is used to identify red signal regions.
2. **Motion detection:** MOG2 background subtraction separates moving objects from the background.
3. **Vehicle detection:** Contour-based processing identifies candidate vehicles.
4. **Stop-line monitoring:** Vehicle-centroid movement is checked against a virtual stop line, including a tilted-line configuration.
5. **Violation detection:** A crossing event is evaluated against the traffic-signal state.
6. **License-plate processing:** Plate localization and OCR are used to extract text for violation logging.

## Key concepts

- HSV colour space and thresholding.
- Background subtraction.
- Contour detection and centroid tracking.
- Geometric line-crossing checks.
- Optical character recognition (OCR).

The design demonstrates how multiple classical computer-vision stages can be connected into a single application pipeline without relying on a deep-learning detector.

## Evaluation

The notebook contains an implementation of the processing pipeline. Quantitative accuracy, precision, recall or OCR accuracy should be included only after they have been measured on an appropriately labelled test video set.

## Technologies

Python, OpenCV, NumPy and OCR tooling used by the notebook.

---

# 6. Legal NLP for Judgment Prediction and Clause Analysis

**Notebook:** `snlp_final.ipynb`

## Objective

This project explores natural language processing (NLP) techniques for analysing legal text in the Indian legal context. It compares traditional machine learning, recurrent neural networks and transformer-based models for text classification.

## Datasets

The notebook references three datasets:

- **NyayaAnumana Transformers Results**
- **PredEx**
- **Indian Bail Judgments**

The notebook loads the datasets, inspects their columns and missing values, and prepares dataset-specific text and label fields.

## Text preprocessing

The NLP pipeline includes:

- Lowercasing.
- Punctuation removal.
- Stop-word processing.
- Lemmatization.
- Conversion of text into numerical representations.

For classical machine learning, the notebook uses TF-IDF features. Neural models use tokenization and padded sequences, while transformer models use their own tokenizers.

## Models explored

### Traditional machine learning

- Logistic Regression.
- Multinomial Naive Bayes.
- Linear Support Vector Machine.
- Decision Tree.
- Additional classical models defined in the notebook.

### Deep learning

- LSTM.
- GRU.
- LSTM and GRU with Word2Vec embeddings.
- LSTM and GRU with GloVe embeddings.

### Transformer-based models

- BERT.
- DistilBERT.

The notebook compares models across datasets and generates evaluation visualizations.

## Recorded results and interpretation

The notebook's LSTM/GRU evaluation reports different results across the three datasets. For example, the recorded GRU result for the first dataset is approximately 99.51% accuracy, while the result for the third dataset is approximately 63.33%.

The Word2Vec-based GRU experiment reports approximately 99.92% accuracy on the first dataset. Such high scores should be interpreted in light of the dataset size, label distribution, split methodology and potential leakage. Results from different datasets are not directly interchangeable.

The notebook also evaluates transformer-based models. Their final comparative performance should be taken from the relevant saved evaluation outputs rather than inferred from the results of the recurrent models.

## Technologies

Python, Pandas, NumPy, NLTK, Scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, Datasets, Gensim, Matplotlib and Seaborn.

---

# 7. Image-Based Wafer Map Pattern Intelligence

**Notebook:** `Wafer_DL_Final.ipynb`

## Objective

This project investigates deep learning for classifying semiconductor wafer maps into defect-pattern categories. Wafer maps represent the spatial distribution of dies across a wafer and can contain patterns associated with manufacturing defects.

The notebook implements an autoencoder-assisted deep learning framework followed by CNN-based classification.

## Dataset

The notebook references the WM-811K wafer-map dataset and loads wafer-map data from a pickle file.

The dataset processing includes examination of wafer-map dimensions, labelled and unlabelled samples, defect patterns and class distributions.

## Methodology

### 1. Dataset exploration

The notebook examines wafer-map dimensions, sample counts and defect-category distributions.

### 2. Map preprocessing

Wafer maps are processed into consistent representations. The implementation includes dimension filtering, resizing and RGB-style encoding of wafer states.

### 3. Convolutional autoencoder

A convolutional autoencoder is trained to encode and reconstruct wafer-map representations. The encoder is then used as part of the downstream data preparation workflow.

### 4. CNN classification

A CNN is trained to classify wafer maps into defect categories. The notebook includes cross-validation experiments and tracks training and evaluation metrics.

### 5. Visual analysis

The notebook includes visualizations of wafer patterns, class distributions, learned filters, feature maps, t-SNE representations, confidence distributions and misclassified examples.

## Defect categories

The recorded classification report includes categories such as:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch
- None / no labelled defect pattern

## Recorded evaluation

The notebook reports approximately **98% overall accuracy** on an evaluation set of 34,590 samples. However, the dataset is strongly imbalanced, with the `none` category making up a large portion of the evaluation data.

The recorded weighted F1-score is approximately 0.98, while the macro F1-score is approximately 0.90. This difference illustrates why class-wise and macro-averaged metrics are important for defect classification: high overall accuracy can coexist with weaker performance on less frequent defect categories.

## Technologies

Python, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib and wafer-map dataset tooling.

---

# Technology Summary

| Area | Technologies |
|---|---|
| Data processing | NumPy, Pandas |
| Classical machine learning | Scikit-learn |
| Computer vision | OpenCV, image-processing techniques |
| Deep learning | TensorFlow, Keras, PyTorch |
| Image classification | CNNs, ResNet50, self-attention |
| Sequential modelling | LSTM, GRU, sensor-based models |
| Natural language processing | NLTK, TF-IDF, Word2Vec, GloVe |
| Transformers | BERT, DistilBERT, Hugging Face |
| Visualizations | Matplotlib, Seaborn |
| Experimentation | Jupyter Notebook, Google Colab |

# Running the Notebooks

These projects were developed as separate experiments. Their environments and data requirements may differ.

## 1. Recommended environment

Use Python with Jupyter Notebook or JupyterLab. Install the dependencies required by the specific notebook rather than assuming every project uses the same environment.

For example:

```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

For TensorFlow-based notebooks:

```bash
pip install tensorflow
```

For PyTorch-based notebooks, install the appropriate PyTorch build for your operating system and hardware. Transformer-based experiments may also require:

```bash
pip install transformers datasets nltk gensim
```

Additional packages may be required by individual notebooks.

## 2. Dataset setup

Some notebooks expect datasets in Google Drive or use external dataset sources. Update dataset paths to match your environment before running them.

For example, a notebook that uses Google Colab and Google Drive may need its data path changed when run locally.

## 3. Execute and inspect

Run the notebook cells in order. Review the preprocessing outputs, training curves, confusion matrices, classification reports and evaluation metrics. A notebook's saved outputs may reflect an earlier execution, so rerunning it can produce different results depending on the environment, random seed and data.

# Important Notes

- These projects are educational and experimental implementations.
- Reported metrics are specific to the recorded notebook experiments and should not be treated as universal performance guarantees.
- Medical image classification results are not a substitute for professional medical diagnosis or clinical validation.
- Legal NLP predictions should not be treated as legal advice or as a replacement for qualified legal judgment.
- Dataset access, redistribution and model licensing remain subject to their respective terms.

## Repository Purpose

This collection documents practical work across computer vision, medical image classification, sensor-based learning, natural language processing and industrial defect analysis. It provides a place to examine different modelling approaches, compare evaluation methods and revisit individual experiments.

