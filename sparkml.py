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

# Step 5: One-Hot Encode each numerical index column individually
type_encoder = OneHotEncoder(inputCol="type_index", outputCol="type_vec")
operator_encoder = OneHotEncoder(inputCol="operator_index", outputCol="operator_vec")
location_encoder = OneHotEncoder(inputCol="location_index", outputCol="location_vec")

# Step 6: Assemble the encoded feature vectors into a single feature column vector
assembler = VectorAssembler(
    inputCols=["type_vec", "operator_vec", "location_vec"],
    outputCol="features"
)

# Apply the transformations sequentially
indexed_df = type_indexer.fit(crashes_df).transform(crashes_df)
indexed_df = operator_indexer.fit(indexed_df).transform(indexed_df)
indexed_df = location_indexer.fit(indexed_df).transform(indexed_df)

# Execute the new individual encoders
encoded_df = type_encoder.transform(indexed_df)
encoded_df = operator_encoder.transform(encoded_df)
encoded_df = location_encoder.transform(encoded_df)

# Assemble everything into the final vector
assembled_df = assembler.transform(encoded_df).select("features", "fat")

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
