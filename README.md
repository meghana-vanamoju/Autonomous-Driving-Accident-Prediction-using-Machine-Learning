# Autonomous-Driving-Accident-Prediction-using-Machine-Learning

# 🚗 Autonomous Driving Accident Prediction using Machine Learning

## 📌 Overview

This project focuses on analyzing driving behavior and predicting road accidents using machine learning techniques.
It utilizes vehicle telemetry and environmental data to identify patterns that contribute to accident occurrence.

---

## 🎯 Objective

To build a machine learning model that predicts whether an accident will occur based on:

* Traffic conditions
* Road characteristics
* Driver information
* Environmental factors

---

## 📊 Dataset

The dataset includes features such as:

* Traffic Density
* Speed Limit
* Number of Vehicles
* Driver Alcohol Consumption
* Driver Age & Experience
* Weather Conditions
* Road Type & Condition
* Time of Day

---

## ⚙️ Data Preprocessing

* Converted categorical data using **one-hot encoding**
* Handled missing values using **mean imputation**
* Ensured target variable is properly formatted

---

## 📈 Exploratory Data Analysis (EDA)

Performed visualization to understand patterns:

* Accident distribution
* Speed vs Accident relationship
* Traffic density vs Accident

---

## 🤖 Machine Learning Model

* Model Used: **Random Forest Classifier**
* Train-Test Split: 80% training, 20% testing

---

## 📊 Results

* **Accuracy:** ~70%
* Model performs well for non-accident cases
* Some limitations in detecting accident cases

### Confusion Matrix

* High accuracy for safe predictions
* Lower recall for accident detection

### Additional Metrics

* Precision, Recall, and F1-score used for better evaluation

---

## 💡 Key Insights

* Higher speed increases accident probability
* Traffic density influences accident occurrence
* Driver-related factors play a significant role

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```
├── main.py
├── dataset.csv
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install pandas seaborn matplotlib scikit-learn
   ```
3. Run the project:

   ```
   python main.py
   ```

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to analyze driving behavior and predict accidents.
It highlights the importance of data-driven approaches in improving road safety.

---

## 🔮 Future Improvements

* Improve model accuracy using advanced algorithms
* Handle class imbalance more effectively
* Use real-time sensor data
* Integrate deep learning models

---

## 👩‍💻 Author

Meghana

---
