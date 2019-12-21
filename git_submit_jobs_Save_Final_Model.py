
# coding: utf-8


import os
import sys
print(sys.argv)
datapath = sys.argv[1]


import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('session1').getOrCreate()
sc = SparkContext.getOrCreate()



from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def downsample(df):
    
    y=df['is_attributed']
    X=df.drop('is_attributed',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    X = pd.concat([X_train, y_train], axis=1)

    download=X[X['is_attributed']==1]
    not_download=X[X['is_attributed']==0]

    not_download_downsampled = resample(not_download,replace = False,  n_samples = len(download), random_state = 1) 

    # combine minority and downsampled majority
    downsampled = pd.concat([not_download_downsampled, download])

    y_train=downsampled['is_attributed']
    X_train=downsampled.drop('is_attributed',axis=1)

    return X_train,X_test,y_train, y_test



def feature_extraction(df_train):
    df_train['dow'] = df_train['click_time'].dt.dayofweek.astype('uint16')
    df_train['doy'] = df_train['click_time'].dt.dayofyear.astype('uint16')
    df_train['hour'] = df_train['click_time'].dt.hour.astype('uint16')
    features_clicks = ['ip', 'app', 'os', 'device']

    for col in features_clicks:
        col_count_dict = dict(df_train[[col]].groupby(col).size().sort_index())
        df_train['{}_clicks'.format(col)] = df_train[col].map(col_count_dict).astype('uint16')

    features_comb_list = [('app', 'device'), ('ip', 'app'), ('app', 'os')]
    for (col_a, col_b) in features_comb_list:
        df = df_train.groupby([col_a, col_b]).size().astype('uint16')
        df = pd.DataFrame(df, columns=['{}_{}_comb_clicks'.format(col_a, col_b)]).reset_index()      
        df_train = df_train.merge(df, how='left', on=[col_a, col_b])
        
    new_features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'dow',
    'doy',
    'ip_clicks',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'ip_app_comb_clicks',
    'app_os_comb_clicks',
    ]
    df=df_train[new_features]

    return df



from pyspark.sql.types import *
import pandas as pd
data = spark.read.csv(datapath,header=True)
col_names = ["ip","app","device","os","channel","click_time","attributed_time","is_attributed"]
data = data.toDF(*col_names)
pand_df = data.select('*').toPandas() 



pand_df['ip']=pand_df['ip'].astype('uint32')
pand_df['app']=pand_df['app'].astype('uint16')
pand_df['device']=pand_df['device'].astype('uint16')
pand_df['os']=pand_df['os'].astype('uint16')
pand_df['channel']=pand_df['channel'].astype('uint16')
pand_df['is_attributed']=pand_df['is_attributed'].astype('uint16')
pand_df['click_time']=pd.to_datetime(pand_df['click_time'])

data1=feature_extraction(pand_df)
data2=pd.concat([data1,pand_df['is_attributed']],axis=1)
X_train,X_test,y_train, y_test=downsample(data2)

train=pd.concat([X_train,y_train],axis=1)
test=pd.concat([X_test,y_test],axis=1)

from pyspark.sql.types import *
col_names = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'dow',
    'doy',
    'ip_clicks',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'ip_app_comb_clicks',
    'app_os_comb_clicks',
    'is_attributed'
    ]
schema = [StructField(x,LongType(),True) for x in col_names]

mySchema = StructType(schema)

train_set = spark.createDataFrame(train,schema=mySchema)
test_set = spark.createDataFrame(test,schema=mySchema)

train_set = train_set.withColumn("is_attributed", train_set["is_attributed"].cast('double'))
test_set = test_set.withColumn("is_attributed", test_set["is_attributed"].cast('double'))

#Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier

features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'dow',
    'doy',
    'ip_clicks',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'ip_app_comb_clicks',
    'app_os_comb_clicks'
    ]
pipeline_stages = []
pipeline_stages.append(VectorAssembler(inputCols=features,outputCol='feature_vector'))
rf = RandomForestClassifier(featuresCol = 'feature_vector',labelCol='is_attributed',numTrees=10,maxBins=500)
pipeline_stages.append(rf)
pipeline = Pipeline(stages=pipeline_stages)

model = pipeline.fit(train_set)
test_output = model.transform(test_set)

test_output_rdd = test_output.rdd
predictionsAndLabels = test_output_rdd.map(lambda x: (x.prediction,x.is_attributed))
metrics1 = MulticlassMetrics(predictionsAndLabels)
metrics2 = BinaryClassificationMetrics(predictionsAndLabels)
print('ROC of random forest model:{}'.format(metrics2.areaUnderROC))
model.write().overwrite().save('Enter URL where the model has to be saved')  ##Todo
