# Necessary Imports
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Building Spark session
sc = SparkContext('local', 'logistic')
spark = SparkSession \
    .builder \
    .appName("Predictive_maintenance_eda") \
    .getOrCreate()

# Importing data from csv file
df = spark.read.csv('gs://source_data_exp/data/exp1_14drivers_14cars_dailyRoutes.csv', header=True)
#df.show()
# Displaying Number of Records
print("Number of Records : ", df.count())
print("---------------------------------------------------------------\n")
print("Displaying the unique Trouble code values")
df.select('TROUBLE_CODES').distinct().collect()
print("---------------------------------------------------------------\n")
# dropping null trouble codes.
df = df.na.drop(subset=["TROUBLE_CODES"])
print("---------------------------------------------------------------\n")
print("Trouble code values after removing nulls")
df.select('TROUBLE_CODES').distinct().collect()
print("---------------------------------------------------------------\n")
# Statistical Description
print("---------------------------------------------------------------\n")
print("Statictical Description")
print("---------------------------------------------------------------\n")
df.describe().show()
print("---------------------------------------------------------------\n")
print("Schema")
df.printSchema()
print("---------------------------------------------------------------\n")
print("Displaying number of nan values for each column")
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
print("---------------------------------------------------------------\n")
print("Displaying number of nan and null values for each column")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
print("---------------------------------------------------------------\n")
# Dropping columns with high number of nan and null values
df=df.drop('EQUIV_RATIO','ENGINE_RUNTIME','SHORT TERM FUEL TRIM BANK 2','FUEL_PRESSURE','LONG TERM FUEL TRIM BANK 2','MAF','AMBIENT_AIR_TEMP','FUEL_LEVEL','BAROMETRIC_PRESSURE(KPA)','VEHICLE_ID','MODEL','MAKE','DTC_NUMBER','AUTOMATIC')
print("Data after column removal")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
print("---------------------------------------------------------------\n")
#df.show()
# Datatype conversion
df = df.withColumn("THROTTLE_POS", regexp_replace(col("THROTTLE_POS"), "%", ""))\
.withColumn("TIMING_ADVANCE", regexp_replace(col("TIMING_ADVANCE"), "%", ""))\
.withColumn("ENGINE_LOAD", regexp_replace(col("ENGINE_LOAD"), "%", ""))
#df.show()
df = df.withColumn("CAR_YEAR",df.CAR_YEAR.cast(IntegerType()))\
.withColumn("ENGINE_COOLANT_TEMP",df.ENGINE_COOLANT_TEMP.cast(IntegerType()))\
.withColumn("ENGINE_RPM",df.ENGINE_RPM.cast(IntegerType()))\
.withColumn("INTAKE_MANIFOLD_PRESSURE",df.INTAKE_MANIFOLD_PRESSURE.cast(IntegerType()))\
.withColumn("AIR_INTAKE_TEMP",df.AIR_INTAKE_TEMP.cast(IntegerType()))\
.withColumn("SPEED",df.SPEED.cast(IntegerType()))\
.withColumn("MIN",df.MIN.cast(IntegerType()))\
.withColumn("HOURS",df.HOURS.cast(IntegerType()))\
.withColumn("DAYS_OF_WEEK",df.DAYS_OF_WEEK.cast(IntegerType()))\
.withColumn("MONTHS",df.MONTHS.cast(IntegerType()))\
.withColumn("YEAR",df.YEAR.cast(IntegerType()))\
.withColumn("ENGINE_POWER",df.ENGINE_POWER.cast(FloatType()))\
.withColumn("THROTTLE_POS", df.THROTTLE_POS.cast(FloatType())/100)\
.withColumn("TIMING_ADVANCE", df.TIMING_ADVANCE.cast(FloatType())/100)\
.withColumn("ENGINE_LOAD", df.ENGINE_LOAD.cast(FloatType())/100)
#df.show()
# Creating spark temp view
df.createOrReplaceTempView("CLEANED_VIEW")
# Big query - Project and dataset details
temp_view_name = "CLEANED_VIEW"
project_id = "avid-airway-395106"
dataset_id = "predictive_maintenance_dataset"
table_id = "exp1_14_drivers"

# Create a BigQuery external table from the temporary view
spark.sql(f"""
    CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{dataset_id}.{table_id}`
    OPTIONS (
        format = 'PARQUET',
        parquet_compression = 'SNAPPY',
        parquet_enable_dictionary_encoding = true
    )
    AS SELECT * FROM {temp_view_name}
""")