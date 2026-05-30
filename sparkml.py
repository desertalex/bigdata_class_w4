from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session with Hive support enabled
spark = SparkSession.builder \
    .appName("MLlib Flight Crashes Fatalities Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load the data from your Hive 'crashes' table into a Spark DataFrame
# We select features to predict 'fat' (fatalities)
crashes_df = spark.sql("SELECT type, operator, location, fat FROM crashes")

# Step 3: Handle null values by dropping rows where vital data is missing
crashes_df = crashes_df.na.drop(subset=["type", "operator", "location", "fat"])

# Step 4: Convert categorical string columns into numerical indices
type_indexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="skip")
operator_indexer = StringIndexer(inputCol="operator", outputCol="operator_index", handleInvalid="skip")
location_indexer = StringIndexer(inputCol="location", outputCol="location_index", handleInvalid="skip")

# Step 5: (Removed OneHotEncoder entirely to avoid Spark version compatibility errors)

# Step 6: Assemble the indexed numeric columns directly into a single feature vector
assembler = VectorAssembler(
    inputCols=["type_index", "operator_index", "location_index"],
    outputCol="features"
)

# Apply the StringIndexers sequentially to the DataFrame
indexed_df = type_indexer.fit(crashes_df).transform(crashes_df)
indexed_df = operator_indexer.fit(indexed_df).transform(indexed_df)
indexed_df = location_indexer.fit(indexed_df).transform(indexed_df)

# Feed the indexed features straight into the VectorAssembler
assembled_df = assembler.transform(indexed_df).select("features", "fat")

# Step 7: Split data into training (70%) and testing (30%) sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 8: Initialize and train the Linear Regression model targeting 'fat'
lr = LinearRegression(labelCol="fat", featuresCol="features")
lr_model = lr.fit(train_data)

# Step 9: Evaluate model performance on the unseen test data
test_results = lr_model.evaluate(test_data)

rmse_val = test_results.rootMeanSquaredError
r2_val = test_results.r2

print(f"Extraction Completed successfully.")
print(f"RMSE: {rmse_val}")
print(f"R^2: {r2_val}")

# ---- Write model performance metrics to HBase via happybase ----

# Format the performance metric payloads (row_key, column_family:column, value)
data = [
    ('crash_model_metrics', 'cf:rmse', str(rmse_val)),
    ('crash_model_metrics', 'cf:r2',   str(r2_val)),
]

# Distributed function to open an HBase connection and save data per RDD partition
def write_to_hbase_partition(partition):
    # Connects to the master container running the HBase Thrift Server
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('my_table')  # Interacts with your defined HBase table
    
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
        
    connection.close()

# Parallelize the local metrics list into an RDD and push to HBase
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 10: Cclose the Spark Session
spark.stop()
