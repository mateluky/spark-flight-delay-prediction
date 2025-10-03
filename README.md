# ✈️ Flight Arrival Delay Prediction with Apache Spark  

*(Big Data Course Project – Universidad Politécnica de Madrid, Master in Digital Innovation – EIT Digital)*  

This project addresses the real-world challenge of predicting **arrival delays of commercial flights** using **Apache Spark** and **machine learning techniques**.  
It was developed as part of the *Big Data* course at UPM, leveraging the well-known **Airline On-Time Performance Dataset** from the [Data Expo 2009](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7).  



## 📁 Project Structure
```
.
├── app.py # Spark application for model loading & testing
├── best_model/ # Trained ML model (exported from notebook)
├── project.ipynb # End-to-end notebook: prep, training, validation
├── Data/ # Input datasets
│ ├── 1991.csv # Training data (sample, 1991)
│ └── test_data.csv # Test data (sample, 1989)
├── Documentation_Report.pdf # Technical documentation & results report
├── requirements.txt # Python dependencies
└── README.md # Project description & usage guide
```


## 📊 Problem Statement
The objective is to predict the **arrival delay (ArrDelay)** of domestic U.S. flights **using only information available at take-off**.  

This involves:
- Exploratory data analysis & preprocessing  
- Feature engineering (temporal and categorical features)  
- Model training and validation using Spark MLlib  
- Model serialization for reuse in production  
- Deployment and evaluation via a Spark application  


## 🔍 Dataset
Source: [Airline On-Time Performance – Data Expo 2009](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7)  

- **Training year:** 1991 (sampled to 1,000 rows)  
- **Testing year:** 1989 (sampled to 1,000 rows)  

> ⚠️ *Note: To conserve computational resources, both datasets were downsampled to 1,000 rows each for local execution.*  




## ❌ Forbidden Variables
To avoid data leakage, variables available **only after landing** were excluded, such as:  
`ArrTime, ActualElapsedTime, AirTime, TaxiIn, Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay`



## 🎯 Target Variable
- **ArrDelay** → Flight arrival delay (in minutes)




## 🧪 ML Pipeline
### Jupyter Notebook (`project.ipynb`)
- Data ingestion & cleaning  
- Feature transformations (e.g., time parsing, distance bucketing)  
- Model training & hyperparameter tuning  
- Evaluation with multiple metrics (RMSE, MAE, R²)  
- Export of best-performing model → `best_model/`

### Spark Application (`app.py`)
- Loads serialized model & test dataset  
- Applies same feature transformations  
- Generates predictions and evaluates performance with regression + business metrics  



## 📈 Evaluation Metrics
- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R² Score** (Goodness-of-fit)  
- **15-min Accuracy** → % predictions within 15 minutes of true delay  
- **Severe Delay Accuracy** → Accuracy for delays > 60 minutes  


## 🏆 Results

Two models were trained and evaluated on the airline delay dataset:

| Model              | RMSE   | MAE   | R²    | 15-min Accuracy | Severe Delay Accuracy |
|--------------------|--------|-------|-------|-----------------|-----------------------|
| Linear Regression  | 14.66  | 8.04  | 0.592 | –               | –                     |
| Decision Tree      | 14.29  | 8.53  | 0.612 | 0.861           | 0.801                 |

**Best Performing Model:**  
The **Decision Tree** achieved the best overall performance:  
- **RMSE:** 14.29  
- **MAE:** 8.53  
- **R²:** 0.612  
- **15-min Accuracy:** 86.1% of predictions were within 15 minutes of the actual delay  
- **Severe Delay Accuracy:** 80.1% accuracy on delays over 60 minutes  

These results highlight the model’s ability to capture complex relationships in the data, particularly for identifying significant delays.


## 🚀 Running the Application
Ensure Apache Spark is installed and execute:  

```bash
spark-submit --master local[*] app.py "[path_to_test_data]"
```

## 📦 Requirements
- Python 3.11.9  
- Apache Spark (PySpark) 3.5.3  
- Jupyter Notebook  

Dependencies are listed in [`requirements.txt`](requirements.txt).  

---

## 📄 Documentation
For further details on the methodology, experiments, and results, refer to:  
[`Documentation_Report.pdf`](Documentation_Report.pdf)

