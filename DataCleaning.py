import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
import statistics as st
from statistics import stdev 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics._regression import mean_squared_error, r2_score
from pprint import pprint
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture as GM

#imported the original non-processed data 
df = pd.read_csv("burritos_df.csv", header =1) # we set the header to 1, because the OG df took the title as the column header

################################# DESCRIPTIVE STATISTICS ##########################################

#### DESCRIPTIVE STATISTICS: EXPLORING OVERALL RATING ######

overallRatingMean = df["overall"].mean()
#print("Mean Rating", overallRatingMean)

overallRatingMedian = df["overall"].median()
#print("Median Rating", overallRatingMedian)

overallMode = df["overall"].mode()
#print ("Overall Modes:", overallMode) 

overallQuantiles = df["overall"].quantile([.1, .25, .5, .75])
#print("Quantile Values for Overall", overallQuantiles)

overallStdev = np.nanstd(df["overall"]) # this method computes the standard deviation, while ignoring NaNs.
# --> This way it returns the stdev, the spread of a distribution, of all non-NaN elements
#print("Standard Deviation of 'overall' is % s " % (overallStdev))

# Here we are replacing the NaN values of the overall columns with the calculated median value of the overall column
df["overall"] = df["overall"].replace(np.nan, overallRatingMedian)
#print(df["overall"].isnull().any().sum())

# histogram of overall rating, shows some skewness towards the right 
df["overall"].hist(bins=10)
mp.xlabel("Overall Rating Score")
mp.ylabel("Amount")
mp.title("Histogram of Overall Rating Score")
mp.show()

# boxplot of overall rating:
mp.boxplot(df["overall"])
mp.ylabel("Overall Rating Score")
mp.title("Boxplot of Overall Rating")
mp.show()

########### DESCRIPTIVE STATISTICS: EXPLORING COST #######

#Mean:
costMean = df["Cost"].mean()
print("Cost Mean", costMean)
# Median:
costMedian = df["Cost"].median()
#print ("Cost Median", costMedian)
# Mode
costMode = df["Cost"].mode()
print ("Cost Mode:", costMode)

# What are the Values for the quantiles for cost
costQuantiles = df["Cost"].quantile([.1, .25, .5, .75])
print("Quantile Values for Cost", costQuantiles)

# Standard Deviation: 
#costStdev = st.stdev(df["Cost"], xbar=costMean) # If we use this the answer will be NaN
costStdev = np.nanstd(df["Cost"]) # this method computes the standard deviation, while ignoring NaNs.
# --> This way it returns the stdev, the spread of a distribution, of all non-NaN elements
print("Standard Deviation of Cost is % s " % (costStdev)) 

df["Cost"].hist(bins=10)
mp.xlabel("Cost")
mp.ylabel("Amount")
mp.title("Histogram of Cost Before Removing Outlier")
mp.show()

# calculating the medians without taking into account the outliers
costMedian = df.loc[df["Cost"] < 24, 'Cost'].median()
print ("Cost Median", costMedian)

# replacing the outliers and NaNs with the median values 
df["Cost"] = np.where(df["Cost"] > 24, costMedian,df["Cost"])
df["Cost"] = df["Cost"].replace(np.nan, costMedian)
#print(df["Cost"].isnull().any().sum())

df["Cost"].hist(bins=10)
mp.xlabel("Cost")
mp.ylabel("Amount")
mp.title("Histogram of Cost After Removing Outlier")
mp.show()

###### DESCRIPTIVE STATISTICS: EXPLORING HUNGER #######

# calculating the median 
hMedian = df['Hunger'].median()
print("Hunger Median:", hMedian)

hMean = df['Hunger'].mean()
print("Hunger Mean:", hMean)

hungerQuantiles = df["Hunger"].quantile([.1, .25, .5, .75])
print("Quantile Values for Hunger", hungerQuantiles)

# replacing the NaNs with the median values 
df["Hunger"] = df["Hunger"].replace(np.nan, hMedian)
# print(df["Hunger"].isnull().any().sum())

hungerStdev = np.nanstd(df["Hunger"]) # this method computes the standard deviation, while ignoring NaNs.
# --> This way it returns the stdev, the spread of a distribution, of all non-NaN elements
print("Standard Deviation of Hunger is % s " % (hungerStdev)) 

hMode = df["Hunger"].mode()
print ("Hunger Mode:", hMode)

# histogram of overall Hunger , shows some skewness towards the right 
df["Hunger"].hist(bins=10)
mp.xlabel("Hunger")
mp.ylabel("Amount")
mp.title("Histogram of Hunger Score")
mp.show()

#### DESCRIPTIVE STATISTICS: EXPLORING FILLING ####

# Checking how many NaN values the Fillings column has 
print(df["Fillings"].isnull().any().sum()) # --> this only gave us one Nan value, so just dropped it
# print(len(df))

dfF= df[df['Fillings'].notna()] 

###### DESCRIPTIVE STATISTICS: EXPLORING YELP AND Google Reviews #######

overallYelpMean = df["Yelp"].mean()
#print(overallYelpMean)
overallYelpMedian = df["Yelp"].median()

overallGoogleMean = df["Google"].mean()
#print(overallGoogleMean)
overallGoogleMedian = df["Google"].median()

######### DESCRIPTIVE STATISTICS: EXPLORING VOLUME #####

# As we will see a few lines below, the Volume column has many NaN values which is why we decided to create a new data frame for Volume
dfVolume= df
print("len df", len(dfVolume))
# dropping the null values below
dfVolume["Volume"] = dfVolume["Volume"].replace(" ", np.nan) # we had to do this, since the Volume column always saw an empty string
dfVolume= dfVolume[dfVolume['Volume'].notna()] # mention in the report that we are reducing it quite a bit, we couldve replaced it with median values as well 
dfVolume= dfVolume[dfVolume['Hunger'].notna()]
print("After dropping nans", len(dfVolume))
#print(dfVolume['Volume'].isnull().any().sum())

vMedian = dfVolume["Volume"].median()
#print("vMedian:",vMedian)

# histogram of Volume  
dfVolume["Volume"].hist(bins=10)
mp.xlabel("Volume")
mp.ylabel("Amount")
mp.title("Histogram of Volume")
mp.show()

#########################################   ALGORITHMS     ####################################################

#### LINEAR REGRESSION MEASURING OVERALL SCORE/ HUNGER ######

oTrain = np.array(df['overall'][:300])
oTest = np.array(df['overall'][300:])

hTrain = np.array(df['Hunger'][:300])
hTest = np.array(df['Hunger'][300:])

model = LinearRegression()
model.fit(oTrain.reshape(-1,1), hTrain.reshape(-1,1))

predictions = model.predict(oTest.reshape(-1,1))

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(hTest, predictions))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(hTest, predictions))
 
mp.scatter(oTest, hTest, color='black')
mp.plot(oTest, predictions, color = "blue", linewidth=3)
mp.xlabel("Overall Score")
mp.ylabel("Hunger")
mp.title("Linear Regression of Score and Hunger")
mp.show()

#### LINEAR REGRESSION MEASURING OVERALL SCORE/ FILLINGS SCORE ######

vTrain = np.array(dfF['Fillings'][:300])
vTest = np.array(dfF['Fillings'][300:])

cTrain = np.array(dfF['overall'][:300])
cTest = np.array(dfF['overall'][300:])

model = LinearRegression()
model.fit(vTrain.reshape(-1,1), cTrain.reshape(-1,1))

predictions = model.predict(vTest.reshape(-1,1))

# # The coefficients
print('Coefficients: \n', model.coef_)
# # The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(cTest, predictions))
# # The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(cTest, predictions))
 
mp.scatter(vTest, cTest, color='black')
mp.plot(vTest, predictions, color = "blue", linewidth=3)
mp.xlabel("Fillings Score")
mp.ylabel("Overall Score")
mp.title("Linear Regression Filling Score and Overall Score")
mp.show()

########## K-MEANS ############

###  K-MEANS for Volume, Hunger ###

dfKM = dfVolume[['Volume', 'Hunger']].copy()

X = dfKM.values
kmeans = KMeans(n_clusters=3, random_state=0, max_iter= 30).fit(X)
y_kmeans = kmeans.predict(X)

mp.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
mp.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
mp.title("K-Means Clustering of Volume and Hunger")
mp.show()

# Elbow method
distortions = []
K = range(1,10)
    
for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plotting the elbow method
mp.plot(K, distortions, 'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title("Elbow method for Volume and Hunger")
mp.show()

############ GAUSIAN MIXTURE MODEL ############

gmm = GM(n_components=3).fit(X)
labels = gmm.predict(X)
mp.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
mp.title("Gausian Mixture Model of Volume and Hunger")
mp.show()

## When the first Kmeans model didn't yield any obvious results, we played around with some other combinations. 
####### K-MEANS for Uniformity and Overall Rating ###

df9 = df[df['Uniformity'].notna()]
#print(df9)
dfKM2 = df9[['Uniformity', 'overall']].copy()

X = dfKM2.values
kmeans = KMeans(n_clusters=3, random_state=0, max_iter= 30).fit(X)
y_kmeans = kmeans.predict(X)

mp.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
mp.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
mp.title("K-Means Clustering of Uniformity and Overall Rating")
mp.show()

# Elbow method
distortions = []
K = range(1,10)
    
for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plotting the elbow method
mp.plot(K, distortions, 'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title("Elbow method for Uniformity and Overall Score")
mp.show()

############ GAUSIAN MIXTURE MODEL ############

gmm = GM(n_components=3).fit(X)
labels = gmm.predict(X)
mp.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
mp.title("Gausian Mixture Model of Uniformity and Overall Score")
mp.show()

##########  K-MEANS for Volume, Cost ############
## next we tried out a kmeans model for volume and cost, with a similar line of thought as to why we tried hunger and volume. 
dfKM3 = dfVolume[['Volume', 'Cost']].copy()

X = dfKM3.values
kmeans = KMeans(n_clusters=3, random_state=0, max_iter= 30).fit(X)
y_kmeans = kmeans.predict(X)

mp.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
mp.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
mp.title("K-Means Clustering of Volume and Cost")
mp.show()

# Elbow method
distortions = []
K = range(1,10)
    
for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plotting the elbow method
mp.plot(K, distortions, 'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title("Elbow method for Volume and Cost")
mp.show()

############ K-MEANS for Hunger and Overall Score ##############

dfKM4 = df[['Hunger', 'overall']].copy()

X = dfKM4.values
kmeans = KMeans(n_clusters=3, random_state=0, max_iter= 30).fit(X)
y_kmeans = kmeans.predict(X)

mp.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
mp.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
mp.title("K-Means Clustering of Hunger and Overall Score")
mp.show()

# Elbow method
distortions = []
K = range(1,10)
    
for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plotting the elbow method
mp.plot(K, distortions, 'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title("Elbow method for Hunger and Overall Score")
mp.show()

############ K-MEANS for Synergy and Overall Score ##############

## we decided to cluster synergy and score as well, when we found out that synergy is our most indicative feature from the decision tree

SynergyMedian = df["Synergy"].median()
df["Synergy"] = df["Synergy"].replace(np.nan, SynergyMedian)

dfKM4 = df[['Synergy', 'overall']].copy()

X = dfKM4.values
kmeans = KMeans(n_clusters=4, random_state=0, max_iter= 30).fit(X)
y_kmeans = kmeans.predict(X)

mp.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
mp.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
mp.title("K-Means Clustering of Synergy and Overall Score")
mp.show()

# Elbow method
distortions = []
K = range(1,10)
    
for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plotting the elbow method
mp.plot(K, distortions, 'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title("Elbow method for Synergy and Overall Score")
mp.show()

# ########## Decisions for value splits in the ID3 ##################

# histogram of overall rating, shows some skewness towards the right 
############ with bin size of 20, we can see the normal distribution is skewed a lot to the right. But we can also see there is a gap around 4, therefore we decide, that a good rated burrito has an overall score of 4 and above.
########### this will be used as target feature in the decisiontree, to figure out, which characteristic of the burrito has the highest influence / information gain

# df["overall"].hist(bins=20) # 4
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Tortilla"].hist(bins=20) # 3.5 
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Meat"].hist(bins=20) # 4
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Temp"].hist(bins=20) # 4 
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Wrap"].hist(bins=20) # 4.5
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Meat:filling"].hist(bins=20) #3.8
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Salsa"].hist(bins=20) # 3.5
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Uniformity"].hist(bins=20) # 4
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# df["Synergy"].hist(bins=20) # 4
# mp.xlabel("Rating Score")
# mp.ylabel("Amount")
# mp.show()

# ################## Decision tree Data Prep######################
#create subset of original data, with only the columns needed
print(df.head())

decisiontreeset = df[["Tortilla", "Temp", "Meat","Meat:filling","Uniformity","Salsa","Synergy","Wrap","overall"]]

# replace null values, with median score for the repsective columns
decisiontreeset["Tortilla"] = decisiontreeset["Tortilla"].replace(np.nan, decisiontreeset["Tortilla"].median())
decisiontreeset["Temp"] = decisiontreeset["Temp"].replace(np.nan, decisiontreeset["Temp"].median())
decisiontreeset["Meat"] = decisiontreeset["Meat"].replace(np.nan, decisiontreeset["Meat"].median())
decisiontreeset["Meat:filling"] = decisiontreeset["Meat:filling"].replace(np.nan, decisiontreeset["Meat:filling"].median())
decisiontreeset["Uniformity"] = decisiontreeset["Uniformity"].replace(np.nan, decisiontreeset["Uniformity"].median())
decisiontreeset["Salsa"] = decisiontreeset["Salsa"].replace(np.nan, decisiontreeset["Salsa"].median())
decisiontreeset["Synergy"] = decisiontreeset["Synergy"].replace(np.nan, decisiontreeset["Synergy"].median())
decisiontreeset["Wrap"] = decisiontreeset["Wrap"].replace(np.nan, decisiontreeset["Wrap"].median())
decisiontreeset["overall"] = decisiontreeset["overall"].replace(np.nan, decisiontreeset["overall"].median())

# creating new binary columns, by the split points decided for each feature.             
decisiontreeset["rating"] = ["good" if x >= 4 else "bad" for x in decisiontreeset['overall']]

decisiontreeset["Tortilla"] = ["high" if x >= 3.5 else "low" for x in decisiontreeset['Tortilla']]
decisiontreeset["Temp"] = ["high" if x >= 4 else "low" for x in decisiontreeset['Temp']]
decisiontreeset["Meat"] = ["high" if x >= 4 else "low" for x in decisiontreeset['Meat']]
decisiontreeset["Meat:filling"] = ["high" if x >= 3.8 else "low" for x in decisiontreeset['Meat:filling']]
decisiontreeset["Uniformity"] = ["high" if x >= 4 else "low" for x in decisiontreeset['Uniformity']]
decisiontreeset["Salsa"] = ["high" if x >= 3.5 else "low" for x in decisiontreeset['Salsa']]
decisiontreeset["Synergy"] = ["high" if x >= 4 else "low" for x in decisiontreeset['Synergy']]
decisiontreeset["Wrap"] = ["high" if x >= 4.5 else "low" for x in decisiontreeset['Wrap']]

print(decisiontreeset.head())

dataset = decisiontreeset
print(dataset.columns[:-2])

########################### Actually Decisions Tree #########################

def entropy(target_col):

      ##only parameter is target_column, which species the target
      #calculates the entropy of the dataset
      elements, counts = np.unique(target_col,return_counts = True)
      entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range (len(elements))])
      
      return entropy

def infoGain(data, split_attribute_name, target_name="rating"):
      #calculate the info gain of a dataset. 
      #target_name should be the name of the target feature
      total_entropy = entropy(data[target_name])

      #calculate the values and the corresponding counts for the split attribute
      vals, counts= np.unique(data[split_attribute_name], return_counts=True)

      #calculate the weighted entropy
      weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

      #calculate the information gain
      information_gain = total_entropy - weighted_entropy
      # print(total_entropy, " : ", weighted_entropy)
      # print(split_attribute_name, " : ", information_gain)
      return information_gain

def ID3(data,originaldata,features,target_attribute_name="rating", parent_node_class = None) :
      # data, the data for which the id3 algorithm should be run
      # originaldata, this is the original dataset needed to calculate 
      # features, the feature space. 
      # target_attribute_name, the name of the target attribute
      # parent_node_class, this is the value or class of the mode target feature of the parent node for a specific node.

      #define the stopping criteria --> if one of this is satisfied, we want to return a leaf node

      #if all target_values have the same value, return this value
      if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
      
      #if the dataset is empy, reutrn the mode target feature value in the original dataset
      elif len(data)==0:
            return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
      
      #if the feature space is empty, return the mode target feature value of the direct parent_node
      #the direct parent node is that node which has called the current run of the id3 and hence
      #the mode target feature value is stored in the parent_node_class variable
      elif len(features) ==4:
            return parent_node_class
      
      # if none of the above holds true, grow the tree

      else:
            #set the default value for this node _> the mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]

            #select the feature which best splits the dataset
            item_values = [infoGain(data,feature,target_attribute_name) for feature in features]

            #return the information gain values for the features in the dataset
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]

            #create teh tree structure. the root get the name of the feature (best_feature) with the maximum information
            #gain in the first run
            tree = {best_feature:{}}

            #remove the feature with the best information gain from the feature space
            features = [i for i in features if i != best_feature]

            #grow a branch under the root node for each possible value of the root node feature

            for value in np.unique(data[best_feature]):
                  value = value
                  #split the dataset along the value of the feature with largest infomration gain and then create sub_datasets
                  sub_data = data.where(data[best_feature] == value).dropna()

                  #call the id3 for each of those sub_datasets with the new parameter ... recursion

                  subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
                  #add the subtree, grown from the sub_dataset to the tree under the root

                  tree[best_feature][value] = subtree
            return(tree) 

def predict(query,tree,default = 1):

      for key in list(query.keys()):
            if key in list(tree.keys()):
                  try:
                        result = tree[key][query[key]]
                  except:
                        return default
                  result = tree[key][query[key]]
                  if isinstance(result,dict):
                        return predict(query,result)
                  else:
                        return result

def train_test_split(dataset):
                  #spilliting our data into 300 instances of traninig and 85 instances for testeting
      training_data = dataset.iloc[:300].reset_index(drop=True)
      testing_data = dataset.iloc[300:].reset_index(drop=True)
      return training_data,testing_data

training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

def test(data,tree):
      queries = data.iloc[:,:-1].to_dict(orient = "records")

      predicted = pd.DataFrame(columns=["predicted"])

      for i in range(len(data)):
            predicted.loc[i,"predicted"] = predict(queries[i], tree, 1.0)
      print("The prediction accruacy is: ", (np.sum(predicted["predicted"]==data["rating"])/len(data))*100,"%")

# running and printing the tree and predictions
tree = ID3(training_data, training_data, dataset.columns[[0,1,2,3,4,5,6,7]])
pprint(tree)
test(testing_data,tree)

##### What is the highest rated Burrito, and where do we get it? :O

df["Burrito"] = df["Burrito"].str.lower()
burritoBar = pd.Series(df.groupby(['Burrito'])['overall'].mean())
print(burritoBar.sort_values(ascending=False).head(20))
print(burritoBar.sort_values(ascending=False).tail(10))

df["Location"] = df["Location"].str.lower()
locationBar = pd.Series(df.groupby(['Location'])['overall'].mean())
print(locationBar.sort_values(ascending = False).head(20))
print(locationBar.sort_values(ascending = False).tail(10))
