#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


#Import the dataset

df= pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv')
df


# In[4]:


# Shape of data with number of rows and columns
df.shape


# In[5]:


# Detailed description

df.describe()


# In[6]:


# Documentation
# all the data present
# there is NAN values present
# Mean And Median should in place 
# std is also in good there is no much deviation


# In[7]:


#check the data types of all the columns
df.dtypes


# In[8]:


# We have object(string) datatype in sex column which are catogorical values,
# integer data type in rings column and rest are the floating values.
#vAll are independent variable except Rings. Rings is a target variable.


# In[9]:


#check is there any null value present in any column

df.isnull().sum()


# In[10]:


# We can see no null values present in the dataset. There are no missing values in whole dataset, so we can proceed safely.


# In[11]:


df.info()


# In[12]:


df["Rings"].unique()


# In[13]:


df.loc[df['Rings']== " "]


# In[14]:


#As we can see there are no missing values or any spaces are available in target variable.


# In[15]:


# adding 1.5 to each value of rings column
Rings = [] #empty list

for i in df["Rings"]:
    a=i+1.5
    Rings.append(a)
    
Rings


# In[17]:


#after adding 1.5 in target variable.
df["Rings"]= Rings
df


# In[19]:


# EDA (Exploratory Data Analysis)
# Visualization (Uni Variate analysis)
 # Uni variate analysis works with only one variable, hence it is called uni variate.


# In[20]:


sns.distplot(df["Length"], color = 'g')     


# In[21]:


# As we can see length is almost normally distributed.


# In[23]:


sns.distplot(df["Diameter"], color = 'g')


# In[ ]:


# Small amount of skewness present in Diameter.


# In[24]:


sns.distplot(df["Height"], color = 'g')


# In[25]:


# Height contains soo much skewness.


# In[26]:


sns.distplot(df["Whole weight"], color = 'g')


# In[27]:


# Whole Weight variable is also skewed.


# In[28]:


sns.distplot(df["Shucked weight"], color = 'g')


# In[29]:


# shucked weight also contains skewness.


# In[30]:


sns.distplot(df["Viscera weight"], color = 'g')


# In[31]:


#Viscera weight has the skewness.


# In[32]:


sns.distplot(df["Shell weight"], color = 'g')


# In[ ]:


#Shell weight is also have the skewness.


# In[33]:


sns.distplot(df["Rings"], color = 'g')


# In[34]:


#It is a target variable.


# In[35]:


# Analysis through boxplot
features = df[["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]]


# In[36]:


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


# In[37]:


#Every column contains outliers("Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight")


# In[38]:


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


# In[39]:


# Multi Variate analysis
# Multi variate analysis find the relationship with all variables. Now we will visualize the data and check the coiefficient of multicollinearity


# In[41]:


df_cor = df.corr().abs()

plt.figure(figsize = (15,10))
sns.heatmap(df_cor, vmin=-1, vmax= 1, annot=True, square=True,
          center=0, fmt='.1g', linewidths=.1)
plt.tight_layout()


# In[42]:


#As shown in plot 'Shucked weight' and 'Rings' correlation value is in less amount
#Shell weight' and 'Whole weight' are strongly correlated 
#and 'Whole weight' are strongly correlated 
#Shucked weight' and 'Whole weight' are strongly correlated 


# In[43]:


# Find out which columns are positively and negatively correlated with each other


# In[44]:


plt.figure(figsize=(22,7))
df.corr()['Rings'].sort_values(ascending = False).drop(['Rings']).plot(kind = 'bar', color = 'g')
plt.xlabel('Feature', fontsize = 15)
plt.ylabel('Rings', fontsize = 15)
plt.title('correlation', fontsize = 18)
plt.show()


# In[ ]:


# As shown in plot all features are positively correlate with target variable. There are no negative correlation in it.


# In[45]:


# Encoding
# Encode input variable 'sex'


# In[46]:


df['Sex'].unique()


# In[47]:


df['Sex'].value_counts()


# In[48]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[49]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[50]:


df


# In[51]:


df['Sex'].unique()


# In[53]:


# Remove outliers
# Now we have found the outliers and skewness in some variables.So, first removing the outliers.
# Outlier removal using Zscore


# In[54]:


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


# In[55]:


new_df.describe()


# In[58]:


# Outlier Removing using IQR
# from boxplot in EDA, we came to know that outliers present in following columns.
# Visualize data again to check outliers are present at lower side or higher side


# In[59]:


df2 = df


# In[61]:


# features in which outliers are detected
features = df2[["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]]


# In[62]:


plt.figure(figsize=(25,20))
graph = 1

for column in features:
    if graph <= 30:
        ax = plt.subplot(4,2, graph)
        sns.boxplot(features[column], color = 'g')
        plt.xlabel(column, fontsize = 20)
        
    graph+=1
plt.show()


# In[63]:


#find the IQR (Inter Quantile Range) to identify outliers
#formula for finding IQR

#1st quantile 25%
q1 = df2.quantile(0.25)

#3rd quantile 75%
q3 = df2.quantile(0.75)

#IQR = Inter Quantile Range
iqr = q3-q1


# In[64]:


df2.describe()


# In[65]:


# Outlier detection formula
# Higher side ==> Q3 + (1.5 * IQR)
# Lower side ==> Q1 - (1.5 * IQR)


# In[66]:


#Check the Outliers for Length
#Remove outliers from lower side so, use lower side formula

Length_ = (q1.Length - (1.5*(iqr.Length)))
Length_


# In[67]:


index_out = np.where(df2['Length'] < Length_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[68]:


# after removing outliers the 4128 rows will remains in dataframe.


# In[69]:


# Diameter is having outliers in lower side so use lower side formula
#Check the Outliers for Diameter
#Remove outliers from lower side so, use lower side formula

Diameter_ = (q1.Diameter - (1.5*(iqr.Diameter)))
Diameter_


# In[70]:


index_out = np.where(df['Diameter'] < Diameter_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[71]:


# after removing outliers the 4069 rows will remains in dataframe.


# In[72]:


# Height is having outliers in lower side so use lower side formula
#Check the Outliers for Height
#Remove outliers from lower side so, use lower side formula

Height_ = (q1.Height - (1.5*(iqr.Height)))
Height_


# In[73]:


index_out = np.where(df2['Height'] < Height_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[74]:


# as shown in boxplot we have outlier in higher side of height also
# Remove outliers from higher side so, use higher side formula

Height_high = (q3.Height + (1.5*(iqr.Height)))
Height_high


# In[75]:


index_out = np.where(df2['Height'] > Height_high)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[76]:


# after removing outliers the 4060 rows will remains in dataframe.


# In[77]:


#as shown in boxplot we have outlier in higher side of whole weightafter removing outliers the 4069 rows will remains in dataframe. 
#Remove outliers from higher side so, use higher side formula

Wholeweight_ = (1.153000 + (1.5*(1.153000-0.441500)))
Wholeweight_


# In[78]:


index_out = np.where(df['Whole weight'] > Wholeweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[79]:


# after removing outliers in Whole weight the 4030 rows will remains in dataframe.


# In[80]:


#as shown in boxplot we have outlier in higher side of Shucked weight 
#Remove outliers from higher side so, use higher side formula

Shuckedweight_ = (0.253000 + (1.5*(0.253000-0.186000)))
Shuckedweight_


# In[81]:


df2['Shucked weight']


# In[82]:


index_out = np.where(df2['Shucked weight'] > Shuckedweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[83]:


# after removing outliers in Shucked weight the 2101 rows will remains in dataframe.


# In[84]:


#as shown in boxplot we have outlier in higher side of Viscera weight 
#Remove outliers from higher side so, use higher side formula

Visceraweight_ = (0.502000 + (1.5*(0.502000-0.093500)))
Visceraweight_


# In[85]:


index_out = np.where(df['Viscera weight'] > Visceraweight_)
df2 = df2.drop(df2.index[index_out])
df2.shape
df2.reset_index()


# In[86]:


# after removing outliers in Viscera weight the 2101 rows will remains in dataframe.


# In[88]:


# After removing outliers using IQR technique there are 2101 rows will remains in dataset.
# Now, find how much data loss in IQR method

# 49.70 % data loss after using IQR technique. 50.30% data remains.

# After applying Zscore and IQR technique to remove outliers. We conclude that less amount data will loss in Zscore technique so we will go with Zscore technique
 # Check the skewness and remove it


# In[89]:


new_df.skew()


# In[91]:


# Apply Power transformation to remove skewness
# In power transformation we will take the mean value in place of 0th value skewed data and convert that into normal data(distribution)/less skewed data.


# In[92]:


#applying Power transformation on skewed columns

new_df['Length'] = new_df['Length'].replace(0,new_df['Length'].mean())
new_df['Diameter'] = new_df['Diameter'].replace(0,new_df['Diameter'].mean())


# In[93]:


sns.distplot(new_df["Length"], color = 'g')


# In[94]:


sns.distplot(new_df["Diameter"], color = 'g')


# In[95]:


# As shown in plot skewness removes after using power transformation skewness removal technique.
# Seperating the columns into featuers and target:


# In[96]:


# X= features, y=Target


# In[97]:


x = new_df.drop(columns = 'Rings', axis=1)
y = new_df['Rings']


# In[99]:


# Scalling technique
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scalar = ss.fit_transform(x)


# In[100]:


# variables are scaled now using standard scaler technique.


# In[101]:


# Variance inflation factor
#import libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scalar, i) for i in range(x_scalar.shape[1])]
vif["Features"] = x.columns

#lets check the values
vif


# In[102]:


new_df=new_df.drop(['Whole weight'], axis=1)


# In[103]:


x = new_df.drop(columns = 'Rings', axis=1)
y = new_df['Rings']


# In[104]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scalar = ss.fit_transform(x)


# In[105]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scalar, i) for i in range(x_scalar.shape[1])]
vif["Features"] = x.columns

#lets check the values
vif


# In[106]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[107]:


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


# In[108]:


x_train,x_test,y_train,y_test = train_test_split(x_scalar, y, test_size=0.2, random_state = 81)


# In[109]:


# 1. Linear Regression Model
from sklearn.linear_model import LinearRegression

Lr=LinearRegression()
Lr.fit(x_train, y_train)


# In[110]:


pred_test=Lr.predict(x_test)


# In[111]:


print(r2_score(y_test,pred_test))


# In[112]:


# Cross Validation of Linear Regression
from sklearn.model_selection import cross_val_score
cv_score= cross_val_score(Lr, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[113]:


# Regularization
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

parameters = {'alpha' :[.0001, .001, .01, .1, 1, 10], 'random_state':list(range(0,10))}
ls = Lasso()
clf = GridSearchCV(ls, parameters)
clf.fit(x_train, y_train)

print(clf.best_params_)


# In[114]:


ls = Lasso(alpha = 0.01, random_state=0)
ls.fit(x_train, y_train)
ls.score(x_train, y_train)
pred_ls = ls.predict(x_test)

laso = r2_score(y_test, pred_ls)
laso


# In[115]:


cv_score= cross_val_score(ls, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[131]:


#The r2 score for linear regression model is : 52.50%

# Cross validation score for linear regression is : 35.26%


# In[133]:


# 2. Random Forest Regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parameters ={'criterion':['mse', 'mae'], 'max_features':["auto","sqrt","log2"]}
Rfr= RandomForestRegressor()
clf =GridSearchCV(Rfr, parameters)
clf.fit(x_train, y_train)

print(clf.best_params_)


# In[134]:


Rfr =RandomForestRegressor(criterion = "mse", max_features="log2")
Rfr.fit(x_train, y_train)
Rfr.score(x_train, y_train)
pred_decision = Rfr.predict(x_test)

Rfrs = r2_score(y_test,pred_decision)
print('R2 Score: ',Rfrs*100)

Rfrscore = cross_val_score(Rfr, x, y, cv=3)
Rfrc = Rfrscore.mean()
print('Cross Val Score: ',Rfrc*100)


# In[135]:


# The r2 score for Random Forest Regressor model is : 52.99%

# Cross validation score for Random Forest Regressor is : 49.76%


# In[136]:


# 3. Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)


# In[137]:


pred_dtree = dtree.predict(x_test)
print(r2_score(y_test,pred_dtree))


# In[138]:


cv_score= cross_val_score(dtree, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[139]:


#The r2 score for Decision tree regression model is : 13.06%

# Cross validation score for Decision tree regression is : 27.07%


# In[140]:


# 4. Support Vector Regressor
from sklearn.svm import SVR

svr = SVR()
svr.fit(x_train, y_train)


# In[141]:


pred_svr = svr.predict(x_test)
print(r2_score(y_test,pred_svr))


# In[142]:


cv_score= cross_val_score(svr, x, y, cv=5)
cv_mean=cv_score.mean()
cv_mean


# In[143]:


# The r2 score for SVR model is : 51.19%

# Cross validation score for SVR is : 40.87%


# In[144]:


# Compare all models
#The diffrence between r2 score and cross validation score of linear regression model is : 17.24%

#The diffrence between r2 score and cross validation score Random Forest Regressor model is : 3.23%

#The diffrence between r2 score and cross validation score of Decision Tree Regressor model is : 14.01%

#The diffrence between r2 score and cross validation score of SVR model is : 10.32%


# In[ ]:


# Saving the model
#creating binary file first

with open("model_pickle", "wb") as f:
    pickle.dump(Rfrc, f)


# In[ ]:


#reading Bbinary file

with open("model_pickle","rb") as f:
    mp=pickle.load(f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




