
# coding: utf-8


import os
import sys
print(sys.argv)
datapath = sys.argv[1]




from pyspark.ml import PipelineModel
model = PipelineModel.load('enter url where model has to be saved')

from sklearn.model_selection import train_test_split

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
    'click_id',
    'ip',
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
    'click_time'
    ]
    df=df_train[new_features]

    return df



from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName('sessionstream').getOrCreate()

data = spark.read.csv(datapath,header=True)

col_names = ["click_id","ip","app","device","os","channel","click_time"]
sp_df = data.toDF(*col_names)
pand_df = sp_df.select('*').toPandas()

pand_df['click_id']=pand_df['click_id'].astype('uint32')
pand_df['ip']=pand_df['ip'].astype('uint16')
pand_df['app']=pand_df['app'].astype('uint16')
pand_df['device']=pand_df['device'].astype('uint16')
pand_df['os']=pand_df['os'].astype('uint16')
pand_df['channel']=pand_df['channel'].astype('uint16')
pand_df['click_time']=pd.to_datetime(pand_df['click_time'])

dataf=feature_extraction(pand_df)



from pyspark.sql.types import *
col_names = [
    'click_id',
    'ip',
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
schema = [StructField(x,LongType(),True) for x in col_names]
schema.append(StructField('click_time',TimestampType(),True))
mySchema = StructType(schema)
data = spark.createDataFrame(dataf,schema=mySchema)

test_output = model.transform(data)
test_output_bq = test_output.drop('feature_vector','rawPrediction','probability')
test_output_bq.write.csv('gs://bucket-adtrack/data/output',header=True)


from google.cloud import bigquery
import subprocess

def saveToBigQuery(sc, output_dataset, output_table, directory):
    files = directory + '/part-*'
    subprocess.check_call(
        'bq load --source_format CSV '
        '--replace '
        '--autodetect '
        '{dataset}.{table} {files}'.format(
            dataset=output_dataset, table=output_table, files=files
        ).split())
    output_path = sc._jvm.org.apache.hadoop.fs.Path(directory)
    output_path.getFileSystem(sc._jsc.hadoopConfiguration()).delete(output_path, True)


directory = 'gs://bucket-adtrack/data/output'
output_dataset = 'bigdata-jy3012:adtrack_project'
output_table_1 = 'Predictions_datastudio'

saveToBigQuery(spark, output_dataset, output_table_1, directory)

