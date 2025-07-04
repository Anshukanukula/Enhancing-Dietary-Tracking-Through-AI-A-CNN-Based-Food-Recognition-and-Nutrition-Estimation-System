# 🍽️ Enhancing Dietary Tracking Through AI  
## CNN-Based Food Recognition and Nutrition Estimation System

![Food Recognition](https://img.shields.io/badge/DeepLearning-ResNet50-blue) ![Nutrition Analysis](https://img.shields.io/badge/Nutrition-AI--Powered-brightgreen) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)

---

## 📌 Overview

This project presents an AI-powered dietary monitoring system that utilizes a **fine-tuned ResNet50 CNN** to classify food images and estimate **real-time nutritional values**. Designed for healthcare, fitness, and diet planning, this tool bridges the gap between **food recognition** and **nutrition analysis**, offering a seamless experience through a **Streamlit web app**.

---

## 💡 Key Features

- 🍱 Food classification across 101 categories (Food-101 dataset)
- 🔍 Accurate nutrition estimation using Spoonacular API and local DB
- 🔄 Portion size adjustments with real-time nutrient recalculation
- 📊 Macronutrient breakdown via interactive visualizations
- 📥 Downloadable JSON reports with full recipe details
- 🌐 Streamlit-based user interface

---


---

## 🔧 Tech Stack

- **Model**: ResNet50 (Transfer Learning, CNN)
- **Frameworks**: PyTorch, Streamlit
- **API**: Spoonacular API for recipe and nutrient info
- **Dataset**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

---

## 📈 Model Performance

| Metric          | Value       |
|----------------|-------------|
| Top-1 Accuracy | 90.14%      |
| Top-5 Accuracy | 96.2%       |
| Inference Time | ~380 ms     |

---

## 🧠 Model Optimizations

| Technique               | Description                                 | Result          |
|------------------------|---------------------------------------------|-----------------|
| Label Smoothing        | Reduces overfitting (ε=0.1)                 | +2.1% accuracy  |
| AdamW Optimizer        | Better convergence                          | Stable training |
| Mixed Precision        | Faster computation                          | Speed boost     |
| Gradient Clipping      | Prevents exploding gradients                | Stability       |

---

## 🌐 Web App Features

- 📸 Upload a food image
- 🤖 View top-5 predictions and confidence
- 🍛 Access recipes and macronutrients per serving
- 📈 Pie charts for carbs, protein, and fats
- 🔄 Adjust serving size dynamically
- 📤 Download JSON report

---

## 🚀 How to Run

> This app uses Python, PyTorch, and Streamlit

### 1. Install Dependencies

```bash
pip install -r requirements.txt
📊 Sample Output
✅ Image Classification
Uploaded: apple pie

Top Prediction: Apple Pie (87.8%)

Alternatives: Cheesecake, Strawberry Cake, etc.

🔢 Nutrition Analysis
Calories: 284 kcal

Protein: 2g

Carbs: 34g

Fat: 15g

Ingredient-level details + serving adjustment

📚 References
Based on and improved over models like FoodieCal, Nutrition5k, and RecipeIS.

Marin et al. (2021) – Recipe1M+ Dataset

Ghosh & Sazonov (2025) – NoisyViT

Thames et al. (2021) – Nutrition5k

Ayon et al. (2021) – FoodieCal

Full list: See report or paper reference section

🔭 Future Enhancements
🎤 Voice-based food input

🌍 Multilingual interface

📱 Mobile + wearable deployment

🔒 Privacy-preserving federated learning

🧾 Expanded food categories (Indian, Asian, etc.)

👨‍💻 Authors
Kanukula Anshu – kanukulaanshu22it@student.vardhaman.org

Battula Bhoomika Naga Sai

Mungi Vverendhra Kumar

Yugandhar Manchala

📜 License
This project is for academic and research purposes. For commercial use, contact the authors.

