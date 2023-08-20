# -*- coding: utf-8 -*-
"""Predictive_maintenance_eda.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EPknzuZvLbqush6tz5I5iFWwFwsO8ioI
"""
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext('local', 'logistic')
spark = SparkSession \
    .builder \
    .appName("Predictive_maintenance_eda") \
    .getOrCreate()

df = spark.read.csv('gs://source_data_exp/data/exp1_14drivers_14cars_dailyRoutes.csv', header=True)

df.show()

df.count()

df.select('TROUBLE_CODES').distinct().collect()

# dropping null trouble codes.
df = df.na.drop(subset=["TROUBLE_CODES"])

df.select('TROUBLE_CODES').distinct().collect()

df.describe().show()

df.printSchema()

df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

df.count()

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

df=df.drop('EQUIV_RATIO','ENGINE_RUNTIME','SHORT TERM FUEL TRIM BANK 2','FUEL_PRESSURE','LONG TERM FUEL TRIM BANK 2','MAF','AMBIENT_AIR_TEMP','FUEL_LEVEL','BAROMETRIC_PRESSURE(KPA)','VEHICLE_ID','MODEL','MAKE','DTC_NUMBER','AUTOMATIC')

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

df.show()

df = df.withColumn("THROTTLE_POS", regexp_replace(col("THROTTLE_POS"), "%", ""))\
.withColumn("TIMING_ADVANCE", regexp_replace(col("TIMING_ADVANCE"), "%", ""))\
.withColumn("ENGINE_LOAD", regexp_replace(col("ENGINE_LOAD"), "%", ""))

df.show()

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

df.show()

df.createOrReplaceTempView("CLEANED_VIEW")
