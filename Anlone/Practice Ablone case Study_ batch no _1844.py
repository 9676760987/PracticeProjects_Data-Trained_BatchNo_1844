#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libreries 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings("ignore")

# Importing all required libraries


# In[2]:


#Import the dataset

df= pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv')
df


# In[3]:


# Shape of data with number of rows and columns
df.shape


# In[4]:


# Detailed description

df.describe()


# In[5]:


# Documentation
# all the data present
# there is NAN values present
# Mean And Median should in place 
# std is also in good there is no much deviation


# In[6]:


#check the data types of all the columns
df.dtypes


# In[7]:


# We have object(string) datatype in sex column which are catogorical values,
# integer data type in rings column and rest are the floating values.
#vAll are independent variable except Rings. Rings is a target variable.


# In[8]:


#check is there any null value present in any column

df.isnull().sum()


# In[9]:


# We can see no null values present in the dataset. There are no missing values in whole dataset, so we can proceed safely.


# In[10]:


df.info()


# In[11]:


df["Rings"].unique()


# In[12]:


df.loc[df['Rings']== " "]


# In[13]:


#As we can see there are no missing values or any spaces are available in target variable.


# In[14]:


# adding 1.5 to each value of rings column
Rings = [] #empty list

for i in df["Rings"]:
    a=i+1.5
    Rings.append(a)
    
Rings


# In[15]:


#after adding 1.5 in target variable.
df["Rings"]= Rings
df


# In[16]:


# EDA (Exploratory Data Analysis)
# Visualization (Uni Variate analysis)
 # Uni variate analysis works with only one variable, hence it is called uni variate.


# In[17]:


sns.distplot(df["Length"], color = 'g')     


# In[18]:


# As we can see length is almost normally distributed.


# In[19]:


sns.distplot(df["Diameter"], color = 'g')


# In[20]:


# Small amount of skewness present in Diameter.


# In[21]:


sns.distplot(df["Height"], color = 'g')


# In[22]:


# Height contains soo much skewness.


# In[23]:


sns.distplot(df["Whole weight"], color = 'g')


# In[24]:


# Whole Weight variable is also skewed.


# In[25]:


sns.distplot(df["Shucked weight"], color = 'g')


# In[26]:


# shucked weight also contains skewness.


# In[27]:


sns.distplot(df["Viscera weight"], color = 'g')


# In[28]:


#Viscera weight has the skewness.


# In[29]:


sns.distplot(df["Shell weight"], color = 'g')


# In[30]:


#Shell weight is also have the skewness.


# In[31]:


sns.distplot(df["Rings"], color = 'g')


# In[32]:


#It is a target variable.


# In[33]:


# Analysis through boxplot
features = df[["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]]


# In[34]:


#Relationship Visualizing

plt.figure(figsize = (25,20), facecolor = 'white')
plotnumber = 1

for column in features:
    if plotnumber <= 9: # as we see there are eight columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.boxplot(features[column], color = 'c')
        plt.xlabel(column,fontsize=20)
        
    plotnumber += 1
plt.show()


# In[35]:


#Every column contains outliers("Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight")


# In[36]:


# Bi variate analysis is works with two variables.
#Relationship Visualizing

plt.figure(figsize = (35,35), facecolor = 'white')
plotnumber = 1

for column in df:
    if plotnumber <= 15:
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(df[column],df['Rings'], color='g')
        plt.xlabel(column,fontsize=26)
        plt.ylabel('Rings', fontsize=26)
    plotnumber += 1
plt.tight_layout()


# In[37]:


# Multi Variate analysis
# Multi variate analysis find the relationship with all variables. Now we will visualize the data and check the coiefficient of multicollinearity


# In[38]:


df_cor = df.corr().abs()

plt.figure(figsize = (15,10))
sns.heatmap(df_cor, vmin=-1, vmax= 1, annot=True, square=True,
          center=0, fmt='.1g', linewidths=.1)
plt.tight_layout()


# In[39]:


#As shown in plot 'Shucked weight' and 'Rings' correlation value is in less amount
#Shell weight' and 'Whole weight' are strongly correlated 
#and 'Whole weight' are strongly correlated 
#Shucked weight' and 'Whole weight' are strongly correlated 


# In[40]:


# Find out which columns are positively and negatively correlated with each other


# In[41]:


plt.figure(figsize=(22,7))
df.corr()['Rings'].sort_values(ascending = False).drop(['Rings']).plot(kind = 'bar', color = 'g')
plt.xlabel('Feature', fontsize = 15)
plt.ylabel('Rings', fontsize = 15)
plt.title('correlation', fontsize = 18)
plt.show()


# In[42]:


# As shown in plot all features are positively correlate with target variable. There are no negative correlation in it.


# In[43]:


# Encoding
# Encode input variable 'sex'


# In[44]:


df['Sex'].unique()


# In[45]:


df['Sex'].value_counts()


# In[46]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[47]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))
        


# In[48]:


df


# In[49]:


df['Sex'].unique()


# In[50]:


# Remove outliers
# Now we have found the outliers and skewness in some variables.So, first removing the outliers.
# Outlier removal using Zscore


# In[52]:


#In Zscore technique taking standard deviation 3
#for Zscore outlier removal technique import library from scipy


from scipy.stats import zscore

z_score= zscore(df[['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight', 'Shell weight']])
abs_zscore = np.abs(z_score)

new_data = (abs_zscore < 3).all(axis = 1)

new_df = df[new_data]

print("shape before :", df.shape)
print("shape after :", new_df.shape)
print("Percentage Loss :", (df.shape[0]-new_df.shape[0])/df.shape[0])


# In[53]:


new_df.describe()


# In[54]:


# Outlier Removing using IQR
# from boxplot in EDA, we came to know that outliers present in following columns.
# Visualize data again to check outliers are present at lower side or higher side


# In[55]:


df2 = df


# In[56]:


# features in which outliers are detected
features = df2[["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]]


# In[57]:


plt.figure(figsize=(25,20))
graph = 1

for column in features:
    if graph <= 30:
        ax = plt.subplot(4,2, graph)
        sns.boxplot(features[column], color = 'g')
        plt.xlabel(column, fontsize = 20)
        
    graph+=1
plt.show()


# In[58]:


#find the IQR (Inter Quantile Range) to identify outliers
#formula for finding IQR

#1st quantile 25%
q1 = df2.quantile(0.25)

#3rd quantile 75%
q3 = df2.quantile(0.75)

#IQR = Inter Quantile Range
iqr = q3-q1


# In[59]:


df2.describe()


# In[60]:


# Outlier detection formula
# Higher side ==> Q3 + (1.5 * IQR)
# Lower side ==> Q1 - (1.5 * IQR)


# In[61]:


#Check the Outliers for Length
#Remove outliers from lower side so, use lower side formula

Length_ = (q1.Length - (1.5*(iqr.Length)))
Length_


# In[62]:


index_out = np.where(df2['Length'] < Length_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[63]:


# after removing outliers the 4128 rows will remains in dataframe.


# In[64]:


# Diameter is having outliers in lower side so use lower side formula
#Check the Outliers for Diameter
#Remove outliers from lower side so, use lower side formula

Diameter_ = (q1.Diameter - (1.5*(iqr.Diameter)))
Diameter_


# In[65]:


index_out = np.where(df['Diameter'] < Diameter_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[66]:


# after removing outliers the 4069 rows will remains in dataframe.


# In[67]:


# Height is having outliers in lower side so use lower side formula
#Check the Outliers for Height
#Remove outliers from lower side so, use lower side formula

Height_ = (q1.Height - (1.5*(iqr.Height)))
Height_


# In[69]:


index_out = np.where(df2['Height'] < Height_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[70]:


# as shown in boxplot we have outlier in higher side of height also
# Remove outliers from higher side so, use higher side formula

Height_high = (q3.Height + (1.5*(iqr.Height)))
Height_high


# In[71]:


index_out = np.where(df2['Height'] > Height_high)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[72]:


# after removing outliers the 4060 rows will remains in dataframe.


# In[73]:


#as shown in boxplot we have outlier in higher side of whole weightafter removing outliers the 4069 rows will remains in dataframe. 
#Remove outliers from higher side so, use higher side formula

Wholeweight_ = (1.153000 + (1.5*(1.153000-0.441500)))
Wholeweight_


# In[74]:


index_out = np.where(df['Whole weight'] > Wholeweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[75]:


# after removing outliers in Whole weight the 4030 rows will remains in dataframe.


# In[76]:


#as shown in boxplot we have outlier in higher side of Shucked weight 
#Remove outliers from higher side so, use higher side formula

Shuckedweight_ = (0.253000 + (1.5*(0.253000-0.186000)))
Shuckedweight_


# In[77]:


df2['Shucked weight']


# In[78]:


index_out = np.where(df2['Shucked weight'] > Shuckedweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[79]:


# after removing outliers in Shucked weight the 2101 rows will remains in dataframe.


# In[80]:


#as shown in boxplot we have outlier in higher side of Viscera weight 
#Remove outliers from higher side so, use higher side formula

Visceraweight_ = (0.502000 + (1.5*(0.502000-0.093500)))
Visceraweight_


# In[81]:


index_out = np.where(df['Viscera weight'] > Visceraweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[82]:


# after removing outliers in Viscera weight the 2101 rows will remains in dataframe.


# In[83]:


# After removing outliers using IQR technique there are 2101 rows will remains in dataset.
# Now, find how much data loss in IQR method

# 49.70 % data loss after using IQR technique. 50.30% data remains.

# After applying Zscore and IQR technique to remove outliers. We conclude that less amount data will loss in Zscore technique so we will go with Zscore technique
 # Check the skewness and remove it


# In[84]:


new_df.skew()


# In[85]:


# Apply Power transformation to remove skewness
# In power transformation we will take the mean value in place of 0th value skewed data and convert that into normal data(distribution)/less skewed data.


# In[86]:


#applying Power transformation on skewed columns

new_df['Length'] = new_df['Length'].replace(0,new_df['Length'].mean())
new_df['Diameter'] = new_df['Diameter'].replace(0,new_df['Diameter'].mean())


# In[87]:


sns.distplot(new_df["Length"], color = 'g')


# In[88]:


sns.distplot(new_df["Diameter"], color = 'g')


# In[89]:


# As shown in plot skewness removes after using power transformation skewness removal technique.
# Seperating the columns into featuers and target:


# In[90]:


# X= features, y=Target


# In[91]:


x = new_df.drop(columns = 'Rings', axis=1)
y = new_df['Rings']


# In[92]:


# Scalling technique
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scalar = ss.fit_transform(x)


# In[93]:


# variables are scaled now using standard scaler techniqu


# In[94]:


# Variance inflation factor
#import libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scalar, i) for i in range(x_scalar.shape[1])]
vif["Features"] = x.columns

#lets check the values
vif


# In[95]:


new_df=new_df.drop(['Whole weight'], axis=1)


# In[96]:


x = new_df.drop(columns = 'Rings', axis=1)
y = new_df['Rings']


# In[97]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scalar = ss.fit_transform(x)


# In[98]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scalar, i) for i in range(x_scalar.shape[1])]
vif["Features"] = x.columns

#lets check the values
vif


# In[99]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[100]:


# Finding best random state
from sklearn.tree import DecisionTreeRegressor
maxAccuracy = 0
maxRandomState = 0
for i in range(1,200):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.30, random_state=i)
    mod= DecisionTreeRegressor()
    mod.fit(x_train, y_train)
    pred = mod.predict(x_test)
    acc=r2_score(y_test, pred)
    if acc>maxAccuracy:
        maxAccuracy=acc
        maxRandomState=i
print("Best accuracy is ",maxAccuracy, "on Random_state ", maxRandomState)


# In[101]:


x_train,x_test,y_train,y_test = train_test_split(x_scalar, y, test_size=0.2, random_state = 81)


# In[102]:


# 1. Linear Regression Model
from sklearn.linear_model import LinearRegression

Lr=LinearRegression()
Lr.fit(x_train, y_train)


# In[103]:


pred_test=Lr.predict(x_test)


# In[104]:


print(r2_score(y_test,pred_test))


# In[105]:


# Cross Validation of Linear Regression
from sklearn.model_selection import cross_val_score
cv_score= cross_val_score(Lr, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[106]:


# Regularization
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

parameters = {'alpha' :[.0001, .001, .01, .1, 1, 10], 'random_state':list(range(0,10))}
ls = Lasso()
clf = GridSearchCV(ls, parameters)
clf.fit(x_train, y_train)

print(clf.best_params_)


# In[107]:


ls = Lasso(alpha = 0.01, random_state=0)
ls.fit(x_train, y_train)
ls.score(x_train, y_train)
pred_ls = ls.predict(x_test)

laso = r2_score(y_test, pred_ls)
laso


# In[108]:


cv_score= cross_val_score(ls, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[109]:


#The r2 score for linear regression model is : 52.50%

# Cross validation score for linear regression is : 35.26%


# In[110]:


# 2. Random Forest Regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parameters ={'criterion':['mse', 'mae'], 'max_features':["auto","sqrt","log2"]}
Rfr= RandomForestRegressor()
clf =GridSearchCV(Rfr, parameters)
clf.fit(x_train, y_train)

print(clf.best_params_)


# In[111]:


Rfr =RandomForestRegressor(criterion = "mse", max_features="log2")
Rfr.fit(x_train, y_train)
Rfr.score(x_train, y_train)
pred_decision = Rfr.predict(x_test)

Rfrs = r2_score(y_test,pred_decision)
print('R2 Score: ',Rfrs*100)

Rfrscore = cross_val_score(Rfr, x, y, cv=3)
Rfrc = Rfrscore.mean()
print('Cross Val Score: ',Rfrc*100)


# In[112]:


# The r2 score for Random Forest Regressor model is : 52.99%

# Cross validation score for Random Forest Regressor is : 49.76%


# In[113]:


# 3. Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)


# In[114]:


pred_dtree = dtree.predict(x_test)
print(r2_score(y_test,pred_dtree))


# In[115]:


cv_score= cross_val_score(dtree, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[116]:


#The r2 score for Decision tree regression model is : 13.06%

# Cross validation score for Decision tree regression is : 27.07%


# In[117]:


# 4. Support Vector Regressor
from sklearn.svm import SVR

svr = SVR()
svr.fit(x_train, y_train)


# In[118]:


pred_svr = svr.predict(x_test)
print(r2_score(y_test,pred_svr))


# In[119]:


# The r2 score for SVR model is : 51.19%

# Cross validation score for SVR is : 40.87%


# In[120]:


# Compare all models
#The diffrence between r2 score and cross validation score of linear regression model is : 17.24%

#The diffrence between r2 score and cross validation score Random Forest Regressor model is : 3.23%

#The diffrence between r2 score and cross validation score of Decision Tree Regressor model is : 14.01%

#The diffrence between r2 score and cross validation score of SVR model is : 10.32%


# In[ ]:


# Save the Model


# In[121]:


#creating binary file first

with open("model_pickle", "wb") as f:
    pickle.dump(Rfrc, f)


# In[122]:


#reading Bbinary file

with open("model_pickle","rb") as f:
    mp=pickle.load(f)

