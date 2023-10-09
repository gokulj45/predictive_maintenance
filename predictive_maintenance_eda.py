#Importing necessary libraries
#Ignoring warnings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

import pickle

from google.cloud import storage

# Necessary Imports
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, types
from pyspark import SparkContext
from operator import attrgetter

# Building Spark session
sc = SparkContext('local', 'logistic')
spark = SparkSession \
    .builder \
    .appName("Predictive_maintenance_eda") \
    .getOrCreate()

# Importing data from csv file
df = spark.read.csv('gs://dataproc_predictive_maintenance/data/exp1_14drivers_14cars_dailyRoutes.csv', header=True)
#df.show()

print("Count",df.count())
print("------------------------------------------")
print("Sanity Check")
df.show()

print("Trouble codes before Null value handling")
print(df.select('TROUBLE_CODES').distinct().collect())
print("------------------------------------------")
# replacing null trouble codes with constant NIL.
df = df.na.fill(value="NIL",subset=["TROUBLE_CODES"])
print("Trouble codes after Null value handling")
print(df.select('TROUBLE_CODES').distinct().collect())
print("------------------------------------------")
df.describe().show()
print("------------------------------------------")
df.printSchema()
print("------------------------------------------")

# Formatting percentage values
df = df.withColumn("FUEL_LEVEL",regexp_replace(col("FUEL_LEVEL"),"%",""))\
.withColumn("ENGINE_LOAD", regexp_replace(col("ENGINE_LOAD"), "%", ""))\
.withColumn("THROTTLE_POS", regexp_replace(col("THROTTLE_POS"), "%", ""))\
.withColumn("TIMING_ADVANCE", regexp_replace(col("TIMING_ADVANCE"), "%", ""))\
.withColumn("EQUIV_RATIO", regexp_replace(col("EQUIV_RATIO"), "%", ""))\
.withColumn("SHORT TERM FUEL TRIM BANK 2", regexp_replace(col("SHORT TERM FUEL TRIM BANK 2"), "%", ""))\
.withColumn("SHORT TERM FUEL TRIM BANK 1", regexp_replace(col("SHORT TERM FUEL TRIM BANK 1"), "%", ""))

# Displaying count of null / nan values
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Imputing missing values/ null values with mean values grouped by id fields
mean_values = df.groupBy("MARK", "MODEL", "CAR_YEAR", "VEHICLE_ID", "MIN", "HOURS", "DAYS_OF_WEEK", "MONTHS", "YEAR").agg(
round(mean("ENGINE_POWER"),1).alias("MEAN_ENGINE_POWER"),
mean("BAROMETRIC_PRESSURE(KPA)").cast("int").alias("MEAN_BAROMETRIC_PRESSURE(KPA)"),
mean("ENGINE_COOLANT_TEMP").cast("int").alias("MEAN_ENGINE_COOLANT_TEMP"),
round(mean("FUEL_LEVEL"),1).alias("MEAN_FUEL_LEVEL"),
round(mean("ENGINE_LOAD"),1).alias("MEAN_ENGINE_LOAD"),
mean("AMBIENT_AIR_TEMP").cast("int").alias("MEAN_AMBIENT_AIR_TEMP"),
mean("ENGINE_RPM").cast("int").alias("MEAN_ENGINE_RPM"),
mean("INTAKE_MANIFOLD_PRESSURE").cast("int").alias("MEAN_INTAKE_MANIFOLD_PRESSURE"),
round(mean("MAF"),2).alias("MEAN_MAF"),
round(mean("LONG TERM FUEL TRIM BANK 2"),2).cast("int").alias("MEAN_LONG TERM FUEL TRIM BANK 2"),
mean("AIR_INTAKE_TEMP").cast("int").alias("MEAN_AIR_INTAKE_TEMP"),
mean("FUEL_PRESSURE").alias("MEAN_FUEL_PRESSURE"),
mean("SPEED").cast("int").alias("MEAN_SPEED"),
round(mean("SHORT TERM FUEL TRIM BANK 2"),2).alias("MEAN_SHORT TERM FUEL TRIM BANK 2"),
round(mean("SHORT TERM FUEL TRIM BANK 1"),2).alias("MEAN_SHORT TERM FUEL TRIM BANK 1"),
mean("ENGINE_RUNTIME").alias("MEAN_ENGINE_RUNTIME"),
mean("THROTTLE_POS").cast("int").alias("MEAN_THROTTLE_POS"),
round(mean("TIMING_ADVANCE"),2).alias("MEAN_TIMING_ADVANCE"),
round(mean("EQUIV_RATIO"),2).alias("MEAN_EQUIV_RATIO")
)

# Joining the mean values back to the original df
df = df.join(mean_values, on=["MARK", "MODEL", "CAR_YEAR", "VEHICLE_ID", "MIN", "HOURS", "DAYS_OF_WEEK", "MONTHS", "YEAR"], how="left")

# Fill missing values with the calculated mean values
df = df.withColumn("ENGINE_POWER",when(col("ENGINE_POWER").isNull(),col("MEAN_ENGINE_POWER")).otherwise(col("ENGINE_POWER")))\
.withColumn("BAROMETRIC_PRESSURE(KPA)",when(col("BAROMETRIC_PRESSURE(KPA)").isNull(),col("MEAN_BAROMETRIC_PRESSURE(KPA)")).otherwise(col("BAROMETRIC_PRESSURE(KPA)")))\
.withColumn("ENGINE_COOLANT_TEMP",when(col("ENGINE_COOLANT_TEMP").isNull(),col("MEAN_ENGINE_COOLANT_TEMP")).otherwise(col("ENGINE_COOLANT_TEMP")))\
.withColumn("FUEL_LEVEL",when(col("FUEL_LEVEL").isNull(),col("MEAN_FUEL_LEVEL")).otherwise(col("FUEL_LEVEL")))\
.withColumn("ENGINE_LOAD",when(col("ENGINE_LOAD").isNull(),col("MEAN_ENGINE_LOAD")).otherwise(col("ENGINE_LOAD")))\
.withColumn("AMBIENT_AIR_TEMP",when(col("AMBIENT_AIR_TEMP").isNull(),col("MEAN_AMBIENT_AIR_TEMP")).otherwise(col("AMBIENT_AIR_TEMP")))\
.withColumn("ENGINE_RPM",when(col("ENGINE_RPM").isNull(),col("MEAN_ENGINE_RPM")).otherwise(col("ENGINE_RPM")))\
.withColumn("INTAKE_MANIFOLD_PRESSURE",when(col("INTAKE_MANIFOLD_PRESSURE").isNull(),col("MEAN_INTAKE_MANIFOLD_PRESSURE")).otherwise(col("INTAKE_MANIFOLD_PRESSURE")))\
.withColumn("MAF",when(col("MAF").isNull(),col("MEAN_MAF")).otherwise(col("MAF")))\
.withColumn("LONG TERM FUEL TRIM BANK 2",when(col("LONG TERM FUEL TRIM BANK 2").isNull(),col("MEAN_LONG TERM FUEL TRIM BANK 2")).otherwise(col("LONG TERM FUEL TRIM BANK 2")))\
.withColumn("AIR_INTAKE_TEMP",when(col("AIR_INTAKE_TEMP").isNull(),col("MEAN_AIR_INTAKE_TEMP")).otherwise(col("AIR_INTAKE_TEMP")))\
.withColumn("FUEL_PRESSURE",when(col("FUEL_PRESSURE").isNull(),col("MEAN_FUEL_PRESSURE")).otherwise(col("FUEL_PRESSURE")))\
.withColumn("SPEED",when(col("SPEED").isNull(),col("MEAN_SPEED")).otherwise(col("SPEED")))\
.withColumn("SHORT TERM FUEL TRIM BANK 2",when(col("SHORT TERM FUEL TRIM BANK 2").isNull(),col("MEAN_SHORT TERM FUEL TRIM BANK 2")).otherwise(col("SHORT TERM FUEL TRIM BANK 2")))\
.withColumn("SHORT TERM FUEL TRIM BANK 1",when(col("SHORT TERM FUEL TRIM BANK 1").isNull(),col("MEAN_SHORT TERM FUEL TRIM BANK 1")).otherwise(col("SHORT TERM FUEL TRIM BANK 1")))\
.withColumn("ENGINE_RUNTIME",when(col("ENGINE_RUNTIME").isNull(),col("MEAN_ENGINE_RUNTIME")).otherwise(col("ENGINE_RUNTIME")))\
.withColumn("THROTTLE_POS",when(col("THROTTLE_POS").isNull(),col("MEAN_THROTTLE_POS")).otherwise(col("THROTTLE_POS")))\
.withColumn("TIMING_ADVANCE",when(col("TIMING_ADVANCE").isNull(),col("MEAN_TIMING_ADVANCE")).otherwise(col("TIMING_ADVANCE")))\
.withColumn("EQUIV_RATIO",when(col("EQUIV_RATIO").isNull(),col("MEAN_EQUIV_RATIO")).otherwise(col("EQUIV_RATIO")))

# Creating a new column "SHORT_TERM_FUEL_TRIM_BANK", based on the conditions
df = df.withColumn("SHORT_TERM_FUEL_TRIM_BANK",
                   when(col("SHORT TERM FUEL TRIM BANK 1").isNotNull() & col("SHORT TERM FUEL TRIM BANK 2").isNotNull(),
                        (col("SHORT TERM FUEL TRIM BANK 1") + col("SHORT TERM FUEL TRIM BANK 2")) / 2)
                   .when(col("SHORT TERM FUEL TRIM BANK 1").isNotNull() & col("SHORT TERM FUEL TRIM BANK 2").isNull(), col("SHORT TERM FUEL TRIM BANK 1"))
                   .when(col("SHORT TERM FUEL TRIM BANK 2").isNotNull() & col("SHORT TERM FUEL TRIM BANK 1").isNull(), col("SHORT TERM FUEL TRIM BANK 2"))
                   .otherwise(col("SHORT TERM FUEL TRIM BANK 2"))
                  )

# Dropping the temporary columns
df = df.drop("MEAN_ENGINE_POWER", "MEAN_BAROMETRIC_PRESSURE(KPA)", "MEAN_ENGINE_COOLANT_TEMP", "MEAN_FUEL_LEVEL", "MEAN_ENGINE_LOAD", "MEAN_AMBIENT_AIR_TEMP", "MEAN_ENGINE_RPM", "MEAN_INTAKE_MANIFOLD_PRESSURE", "MEAN_LONG TERM FUEL TRIM BANK 2" ,"MEAN_MAF", "MEAN_AIR_INTAKE_TEMP", "MEAN_FUEL_PRESSURE", "MEAN_SPEED", "MEAN_SHORT TERM FUEL TRIM BANK 2", "MEAN_SHORT TERM FUEL TRIM BANK 1", "MEAN_ENGINE_RUNTIME", "MEAN_THROTTLE_POS", "MEAN_DTC_NUMBER", "MEAN_TIMING_ADVANCE", "MEAN_EQUIV_RATIO")

# Show the final DataFrame
df.show()

# Count of null / nan values after imputation
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Dropping columns with large quantity of null / missing values and id columns
df = df.drop('BAROMETRIC_PRESSURE(KPA)','EQUIV_RATIO','MODEL','MAKE','VEHICLE_ID','LONG TERM FUEL TRIM BANK 2','FUEL_LEVEL','AMBIENT_AIR_TEMP','DTC_NUMBER','AUTOMATIC','FUEL_PRESSURE','FUEL_TYPE','SHORT TERM FUEL TRIM BANK 2','SHORT TERM FUEL TRIM BANK 1','INTAKE_MANIFOLD_PRESSURE','MAF')
df = df.na.drop(subset=["ENGINE_RPM","ENGINE_POWER","SHORT_TERM_FUEL_TRIM_BANK","YEAR"])
df = df.drop('MARK', 'CAR_YEAR', 'MIN', 'HOURS', 'DAYS_OF_WEEK', 'MONTHS', 'YEAR', 'TIMESTAMP','ENGINE_RUNTIME')

# Count of nan / null values after preprocessing
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Removing duplicates
df = df.select("*").distinct()

df.count()

df.show()

df.write.option("header",True).mode("overwrite").csv("imputeddata")

# Type Casting
df = df.withColumn("ENGINE_COOLANT_TEMP",df.ENGINE_COOLANT_TEMP.cast(IntegerType()))\
.withColumn("ENGINE_RPM",df.ENGINE_RPM.cast(IntegerType()))\
.withColumn("AIR_INTAKE_TEMP",df.AIR_INTAKE_TEMP.cast(IntegerType()))\
.withColumn("SPEED",df.SPEED.cast(IntegerType()))\
.withColumn("ENGINE_POWER",df.ENGINE_POWER.cast(FloatType()))\
.withColumn("THROTTLE_POS", df.THROTTLE_POS.cast(FloatType())/100)\
.withColumn("TIMING_ADVANCE", df.TIMING_ADVANCE.cast(FloatType())/100)\
.withColumn("ENGINE_LOAD", df.ENGINE_LOAD.cast(FloatType())/100)\
.withColumn("SHORT_TERM_FUEL_TRIM_BANK", df["SHORT_TERM_FUEL_TRIM_BANK"].cast(IntegerType()))\
.withColumn("TROUBLE_CODES",df["TROUBLE_CODES"].cast(StringType()))

# Interpolation
pd_df = df.toPandas()
pd_df = pd_df.interpolate()

# Heatmap
plt.figure(figsize = (15,5))
sns.heatmap(pd_df.iloc[:,1:].corr(),annot=True)

# Histogram
hist = pd_df.hist(bins=24, figsize=(20,20) )

# Encoding trouble codes
Encoder = LabelEncoder()
X=pd_df.drop(columns=['TROUBLE_CODES'])
y=pd_df['TROUBLE_CODES']
# Encoding
y_encoded = Encoder.fit_transform(y)

# Test train split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=20)
print("Test and Train Data Dimension")
print("Shape of Test Dataset - X : ", X_val.shape)
print("Shape of Test Dataset - Y : ", y_val.shape)
print("Shape of Training Dataset - X : ", X_train.shape)
print("Shape of Training Dataset - Y : ", y_train.shape)

# Outlier analysis
#function to display boxplot for features
def drawBoxPlot(data,col,fig,newXname):
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  ax = fig.add_subplot(1, 1, col)
  ax.set_xlabel('xlabel')
  ax.set_ylabel('Ylabel')
  ax.set_title('axes title')

  sns.boxplot(x="features", y="value", hue="TROUBLE_CODES", data=data,ax=ax)
  plt.xticks(rotation=90)
  ax.set_xlabel(newXname + ' Features')
  ax.set_ylabel(newXname + ' Values')
  ax.set_title(newXname + ' Box Plot')

fig1 = plt.figure(figsize=(40,15))
mean_val =  pd.Series(pd_df.all())
data = pd.melt(pd_df,id_vars=["TROUBLE_CODES"],
                    var_name="features",
                    value_name='value')
drawBoxPlot(data,1,fig1,'')

mean_val

# Normalizing the data - Using Robust Scalar
xplt = X_train.copy()
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_plot = np.array(xplt)
X_features = ['ENGINE_POWER', 'ENGINE_COOLANT_TEMP', 'ENGINE_LOAD', 'ENGINE_RPM', 'AIR_INTAKE_TEMP', 'SPEED', 'THROTTLE_POS', 'TIMING_ADVANCE', 'SHORT_TERM_FUEL_TRIM_BANK']
df_normalized = pd.DataFrame(X_train, columns =X_features)

#Functions to draw histogram of original and normalized data
def norm_plot(ax, data):
    scale = (np.max(data) - np.min(data)) * 0.2
    x = np.linspace(np.min(data) - scale, np.max(data) + scale, 50)
    _, bins, _ = ax.hist(data, x, color='xkcd:azure')
    mu = np.mean(data)
    std = np.std(data)
    dist = stats.norm.pdf(bins, loc=mu, scale=std)
    axr = ax.twinx()
    axr.plot(bins, dist, color='orangered', lw=2)
    axr.set_ylim(bottom=0)
    axr.axis('off')

def show_histogram(X_f,X_p,X_p_n):
  fig,ax=plt.subplots(1, 9, figsize=(26, 3))
  for i in range(len(ax)):
      norm_plot(ax[i],X_p[:,i],)
      ax[i].set_xlabel(X_f[i])
  ax[0].set_ylabel("count");
  fig.suptitle("distribution of features before normalization")
  plt.show()
  fig,ax=plt.subplots(1,9,figsize=(26,3))
  for i in range(len(ax)):
      norm_plot(ax[i],X_p_n[:,i],)
      ax[i].set_xlabel(X_f[i])
  ax[0].set_ylabel("count");
  fig.suptitle("distribution of features after normalization")
  plt.show()

# Data Distribution - Histogram Analysis
X_features = ['ENGINE_POWER', 'ENGINE_COOLANT_TEMP', 'ENGINE_LOAD', 'ENGINE_RPM', 'AIR_INTAKE_TEMP', 'SPEED', 'THROTTLE_POS','TIMING_ADVANCE', 'SHORT_TERM_FUEL_TRIM_BANK']
show_histogram(X_features[:],X_plot[:,:],X_train[:,:])

# Initialize a GCS client
client = storage.Client()

# Define your GCS bucket and destination blob path
bucket_name = 'dataproc_predictive_maintenance'
dt_blob_name = 'models/dt/decision_tree_model.sav'
rf_blob_name = 'models/rf/random_forest_model.sav'
knn_blob_name = 'models/knn/knn_classifier_model.sav'
svm_blob_name = 'models/svm/svm_classifier_model.sav'
nb_blob_name = 'models/nb/naive_bayes_model.sav'


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_val)
pickle.dump(decision_tree, open("decision_tree_model", 'wb'))

# Upload the model to GCS
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(dt_blob_name)
blob.upload_from_filename('decision_tree_model')

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_val)
pickle.dump(random_forest, open("random_forest_model", 'wb'))

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(rf_blob_name)
blob.upload_from_filename('random_forest_model')

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_val)
pickle.dump(knn_classifier, open("knn_classifier_model", 'wb'))

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(knn_blob_name)
blob.upload_from_filename('knn_classifier_model')

svm_classifier = SVC(kernel='linear', random_state=1)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_val)
pickle.dump(svm_classifier, open("svm_classifier_model", 'wb'))

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(svm_blob_name)
blob.upload_from_filename('svm_classifier_model')

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_naive_bayes = naive_bayes.predict(X_val)
pickle.dump(naive_bayes, open("naive_bayes_model", 'wb'))

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(nb_blob_name)
blob.upload_from_filename('naive_bayes_model')

# Evaluate the models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("---------------------------")


evaluate_model(y_val, y_pred_decision_tree, "Decision Tree")
evaluate_model(y_val, y_pred_random_forest, "Random Forest")
evaluate_model(y_val, y_pred_svm, "Support Vector Machine (SVM)")
evaluate_model(y_val, y_pred_knn, "K-Nearest Neighbors (KNN)")
evaluate_model(y_val, y_pred_naive_bayes, "Naive Bayes")
