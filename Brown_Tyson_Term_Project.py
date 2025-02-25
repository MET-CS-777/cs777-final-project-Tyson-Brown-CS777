
import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
import time
import kagglehub
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import RegressionEvaluator


def recommender(predictions, beer_name, num_recommendations=10):
    #locates the first cluster where the beer is, potentially problematic as beers can be in multiple clusters
    beer_cluster = predictions.filter(predictions["beer_name"] == beer_name).first()["prediction"]
    #find the other beers in the cluster that are not the same beer
    similar_beers = (predictions.filter(predictions["prediction"] == beer_cluster).filter(predictions["beer_name"] != beer_name)
    #select 10 recommendations.
    .select("beer_name").distinct().limit(num_recommendations)
    .rdd.flatMap(lambda x: x).collect())
    return similar_beers


def user_recommendations(user_recs, indexed_df, reviewer_id):
    # join the recommendations with the main data to get the beer names and styles
    user_rec = user_recs.filter(user_recs.reviewer_id == reviewer_id)
    user_rec = user_rec.join(indexed_df, user_rec.beerId == indexed_df.beer_beerid, "left")\
    .select("beer_name", "beer_style").distinct().rdd.map(lambda x: (x[0], x[1])).collect()
    print("Top 10 recommendations for reviewer with id " + str(reviewer_id))
    return user_rec


# Download latest version
file = kagglehub.dataset_download("rdoume/beerreviews")
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(file, header=True, inferSchema=True)

print(df.show())

# First Idea - implement recommender based on cluster and then match beers with clusters.
# choose the numeric properties of the review to create features,
# things such as beer style and name would be expected to cluster together so those are excluded
#to create a model based purely on reviews.
columns = ["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"]
assembler = VectorAssembler(
    inputCols= columns,
    outputCol="features")

data = assembler.transform(df)
data.cache()

#Find the optimum number of clusters
#optimize k by calculating cost and plot using elbow method
sse_list = []
for k in range(2, 11):
    kmeans = KMeans(k=k, seed=1)
    #train kmeans model
    model = kmeans.fit(data)
    evaluator = ClusteringEvaluator()
    predictions = model.transform(data)
    sse = evaluator.evaluate(predictions)
    sse_list.append(sse)
    print(f"k={k}, SSE={sse}")

plt.plot(range(2, 11), sse_list, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within Set Sum of Squared Errors (SSE)')
plt.xticks(range(1, 10))
plt.grid(True)
plt.show()

#Optimum amount of clusters is 5. Rerun model with 5 clusters.
num_clusters = 5
kmeans = KMeans(k=num_clusters, seed=1)
model = kmeans.fit(data)
predictions = model.transform(data)

# Trying to create clusters based upon beer name is problematic as various users rate the beer differently and
# therefore the beer names appear in various clusters, so trying to make a recommender using this approach probably will not work.
# choose beer that you want similar taste recommendations for.
beer_list = df.select("beer_name").distinct().rdd.flatMap(lambda x: x).collect()  
print("Select from the following list to test: Oktoberfestbier, Balashi, Ludwig's Red Anvil, Basement Bitter")
user_input = input("Enter a beer name: ")
if user_input in beer_list:
    beer_recommender = user_input
    recommendations = recommender(predictions, beer_recommender)
    print(f"Recommendations for {beer_recommender}: {recommendations}")

else:
    print("Enter a valid beer name")


# This model has some limitations due to different reviewer perspectives and therefore beers can have
# multiple cluster assignments. So next will try an ALS model based on total rating.

#fill null values in review_profilename column for indexing
df = df.na.fill({"review_profilename": "unknown"}) 
#df review_profilename must be converted to numerical values instead of strings
indexer = StringIndexer(inputCol="review_profilename", outputCol="reviewer_id")
indexed_df = indexer.fit(df).transform(df)
indexed_df = indexed_df.withColumn("reviewer_id", indexed_df["reviewer_id"].cast(IntegerType()))
indexed_df.cache()

# Split the data into training and test sets
(training_data, test_data) = indexed_df.randomSplit([0.8, 0.2], seed=1234)

# Create an ALS model
als = ALS(maxIter=10, regParam=0.01, userCol="reviewer_id", itemCol="beer_beerid", 
          ratingCol="review_overall", coldStartStrategy="drop")

# Fit the model to the training data
model = als.fit(training_data)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="review_overall",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate recommendations for all users
user_recs = model.recommendForAllUsers(10)  # Generate top 10 recommendations for each user
# convert the recommendations to multiple rows per user with one recommendation in each row
user_recs = user_recs.selectExpr("reviewer_id", "explode(recommendations) as recommendations")
# convert the recommendations column from {beer_id, rating} to tow columns beer_id  and rating
user_recs = user_recs.selectExpr("reviewer_id", "recommendations.beer_beerid as beerId",
                                "recommendations.rating as rating")


reviewer_id = int(input("Enter a number between 1 and 33388: "))

try:
  print(user_recommendations(user_recs, indexed_df, reviewer_id))

except:
  print("user invalid, please try again")
