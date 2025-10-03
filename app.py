## 1. Imports and Setup
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, abs as F_abs, floor
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
import sys

## 2. Spark Session Creation
def create_spark_session():
    """Creates and configures Spark session with 4GB driver memory"""
    return SparkSession.builder \
        .appName("FlightDelayPrediction") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

## 3. Data Preprocessing
def preprocess_data(df):
    """
    Preprocesses the input dataframe with following steps:
    1. Handles missing values
    2. Removes forbidden columns
    3. Filters canceled flights
    4. Creates time-based features
    5. Creates categorical features
    6. Calculates route frequencies
    """    
    # Handle missing values first
    df = df.na.drop(subset=["ArrDelay", "DepDelay", "DepTime"])
    
    # Remove forbidden columns if they exist
    forbidden_cols = [
        "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"
    ]
    existing_forbidden_cols = [col for col in forbidden_cols if col in df.columns]
    if existing_forbidden_cols:
        df = df.drop(*existing_forbidden_cols)

    # Filter out canceled flights first
    df = df.filter(col("Cancelled") == 0) \
           .drop("CancellationCode", "Cancelled") \
           .filter(col("CRSElapsedTime") > 0) \
           .distinct()
    
    # Convert numeric columns to double type
    numeric_cols = ["ArrDelay", "DepDelay", "Distance", "CRSElapsedTime"]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
    
    # Create time-based features
    df = df.withColumn("DepTime_Hour", floor(col("DepTime")/100)) \
        .withColumn("DepTime_Minute", col("DepTime") % 100) \
        .withColumn("CRSDepTime_Hour", floor(col("CRSDepTime")/100)) \
        .withColumn("CRSDepTime_Minute", col("CRSDepTime") % 100) \
        .withColumn("CRSArrTime_Hour", floor(col("CRSArrTime")/100)) \
        .withColumn("CRSArrTime_Minute", col("CRSArrTime") % 100)
    
    # Create categorical features
    df = df.withColumn("IsWeekend", when(col("DayOfWeek").isin([6, 7]), 1).otherwise(0)) \
        .withColumn("TimeOfDay",
            when((col("DepTime_Hour") >= 5) & (col("DepTime_Hour") < 12), "morning")
            .when((col("DepTime_Hour") >= 12) & (col("DepTime_Hour") < 17), "afternoon")
            .when((col("DepTime_Hour") >= 17) & (col("DepTime_Hour") < 22), "evening")
            .otherwise("night"))
    
    # Create distance category
    df = df.withColumn("DistanceCategory",
        when(col("Distance") <= 750, "short")
        .when((col("Distance") > 750) & (col("Distance") <= 2500), "medium")
        .otherwise("long"))
    
    # Calculate route frequencies
    route_frequencies = df.groupBy("Origin", "Dest") \
        .count() \
        .withColumnRenamed("count", "RouteFrequency")
    
    return df.join(route_frequencies, ["Origin", "Dest"])

## 4. Model Evaluation Functions
def evaluate_predictions(predictions):
    """
    Evaluates model predictions using multiple metrics:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - R2 (R-squared score)
    - 15-minute accuracy
    - Severe delay accuracy
    """

    # Filter out null predictions
    valid_predictions = predictions.filter(
        col("ArrDelay").isNotNull() & 
        col("prediction").isNotNull()
    )
    
    # Calculate regression metrics
    evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction")
    
    metrics = {
        "RMSE": evaluator.setMetricName("rmse").evaluate(valid_predictions),
        "MAE": evaluator.setMetricName("mae").evaluate(valid_predictions),
        "R2": evaluator.setMetricName("r2").evaluate(valid_predictions)
    }
    
    # Calculate business metrics
    total = valid_predictions.count()
    accurate_predictions = valid_predictions.filter(
        F_abs(col("prediction") - col("ArrDelay")) <= 15
    ).count()
    
    # Calculate accuracy for severe delays (>60 minutes)
    severe_delays = valid_predictions.filter(col("ArrDelay") > 60)
    if severe_delays.count() > 0:
        severe_correct = severe_delays.filter(col("prediction") > 60).count()
        metrics["Severe_Delay_Accuracy"] = severe_correct / severe_delays.count()
    else:
        metrics["Severe_Delay_Accuracy"] = 0.0
    
    metrics["15min_Accuracy"] = accurate_predictions / total if total > 0 else 0
    
    return metrics

def show_best_worst_predictions(predictions, n=5):
    valid_predictions = predictions.filter(
        col("ArrDelay").isNotNull() & 
        col("prediction").isNotNull()
    )
    
    predictions_with_error = valid_predictions.withColumn(
        "abs_error", 
        F_abs(col("prediction") - col("ArrDelay"))
    )
    
    # Show best predictions
    print("\nBest Predictions (Smallest Error):")
    print("-" * 50)
    predictions_with_error.orderBy("abs_error").select(
        "ArrDelay", 
        "prediction", 
        "abs_error"
    ).show(n)
    
    # Show worst predictions
    print("\nWorst Predictions (Largest Error):")
    print("-" * 50)
    predictions_with_error.orderBy(col("abs_error").desc()).select(
        "ArrDelay", 
        "prediction", 
        "abs_error"
    ).show(n)

## 5. Main Application Logic
def main():
    """Main application entry point"""
    # Validate command line arguments
    if len(sys.argv) != 2:
        print("Usage: spark-submit app.py <test_data_path>")
        sys.exit(1)

    test_data_path = sys.argv[1]
    spark = create_spark_session()
    
    try:
        # Load and process test data
        print("\nLoading test data...")
        df = spark.read.csv(test_data_path, header=True, inferSchema=True)
        print(f"Loaded {df.count()} records")
        
        print("\nPreprocessing data...")
        test_data = preprocess_data(df)
        
        # Load and apply model
        print("\nLoading best model...")
        model = PipelineModel.load("best_model")
        
        print("\nMaking predictions...")
        predictions = model.transform(test_data)
        
        # Evaluate and display results
        metrics = evaluate_predictions(predictions)
        show_best_worst_predictions(predictions)
        
        def get_model_type(model):
            """Gets the type of the last stage in the pipeline"""
            last_stage = model.stages[-1]
            return last_stage.__class__.__name__

        # Display performance metrics
        print("\nModel Performance Metrics:")
        print("-" * 50)
        print(f"Model Type: {get_model_type(model)}")  # Dynamically get model type
        print("-" * 50)
        print(f"RMSE: {metrics['RMSE']:.2f} (Shows the average forecast error in minutes.)")
        print(f"MAE: {metrics['MAE']:.2f} (Average absolute forecast error in minutes.)")
        print(f"R^2 Score: {metrics['R2']:.3f} (Shows the proportion of variance explained by the model. [0-1])")
        print(f"15-minute Accuracy: {metrics['15min_Accuracy']*100:.1f}% (The percentage of accurate predictions within 15 minutes.)")
        print(f"Severe Delay Accuracy: {metrics['Severe_Delay_Accuracy']*100:.1f}% (Forecast accuracy for severe delays [more than 60 minutes])")

        
        print("\nApplication completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()
        print("\nSpark session stopped")

if __name__ == "__main__":
    main()