import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## 1.1 Data description
# We have a dataset with house pricing in Boston in 1970-1980th.
# 
# **CRIM** - per capita crime rate by town  
# **ZN** - proportion of residential land zoned for lots over 25,000 sq.ft.  
# **INDUS** - proportion of non-retail business acres per town.  
# **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
# **NOX** - nitric oxides concentration (parts per 10 million)  
# **RM** - average number of rooms per dwelling  
# **AGE** - proportion of owner-occupied units built prior to 1940  
# **DIS** - weighted distances to five Boston employment centres  
# **RAD** - index of accessibility to radial highways  
# **TAX** - full-value property-tax rate per 10k dollars  
# **PTRATIO** - pupil-teacher ratio by town
# **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
# **LSTAT** -  percentage of lower status of the population  
# **MEDV** - Median value of owner-occupied homes in 1k's of dollars  

# In[63]:


data = pd.read_csv('BostonHousing.csv')


# In[64]:


data.info()


# **13 features, 506 entries, no nulls. Target - medv**

# In[65]:


print(data.describe())


# In[66]:


data.corr()


# In[67]:


sns.clustermap(data.corr())


# Here we can see, that a number of features are correlated to target -**medv** (which is good), and  a number of features are correlated between themselves (which is not so good). Still, we have to do full linear model first.

# ## 1.2 Data standartization and model

# Even though **chas** is a categorical feature, we will standartize it too in order to fit same range as other features (I found online pros and cons for it). 

# In[68]:


features, target = data.iloc[:,:-1], data['medv']
means = features.mean(axis=0)
stds = features.std(axis=0)

features = (features - means) / stds

X = sm.add_constant(features)
model = sm.OLS(target, X)
prediction = model.fit()

print(prediction.summary())


# # 2. Testing linear regression asumptions
# ## 2.1 Linearity

# In[69]:


target_pred = prediction.get_prediction(X).predicted_mean


# In[70]:


slope, intercept = np.polyfit(target, target_pred, 1)
best_line = slope * target + intercept
plt.figure(figsize=(8,8))
sns.scatterplot(target, target_pred, color='purple')
plt.plot(target, best_line, color='green')
plt.title('Relationship between target and prediction', size=15)
plt.xlabel('Target', size=12)
plt.ylabel('Prediction', size=12)


# Here we can see - linearity isn't perfect. Also there is an abnormal number of 50K-priced houses. It looks like aggregation of houses that are priced >=50K. This houses can be dropped for model optimization

# ## 2.2 No outliers

# Let's see, if model is affected by too impactfull otliers. We will use Cook's Distance

# In[71]:


influence = prediction.get_influence()
cooks = influence.cooks_distance
print((cooks[1] < 0.05).sum())
print(sorted(cooks[1])[:10])


# It looks like there are no significant outliers.

# ## 2.3 No multicollinearity

# To measure severity of multicollinearity, we will use variance inflation factor (VIF).

# In[72]:


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                            for i in range(len(X.columns))]
    return vif_data


# In[73]:


calculate_vif(X)


# Here we can see, that two features have VIF>5, so drop of at least one of them can improve interpretability. Though, some sources imply that any VIF under 10 is still all-right.

# ## 2.4 Homoscedasticity 

# In[74]:


resids = target - target_pred
plt.figure(figsize=(8,6), dpi=300)
sns.scatterplot(target_pred, resids, color='purple')
plt.title('Relationship between prediction and residuals', size=15)
plt.xlabel("Prediction", size=12)
plt.ylabel("Residual", size=12);


# Possibly it would look like homoscedasticity if not the houses priced 50k.

# ## 2.5 Distribution of residuals

# In[75]:


plt.figure(figsize=(8,6), dpi=300)
sns.histplot(resids, color='purple')
plt.title('Distribution of residuals', size=15)
plt.xlabel("Residual", size=12)
plt.ylabel("Count", size=12);


# In[76]:


sm.qqplot(resids, fit=True, line ='45', markerfacecolor='purple', alpha=0.2)
plt.title('Distribution of residuals', size=15)


# We can see that distribution of residuals is not so normal. Distribution is somewhat skewed. Effect of 50k houses is seen to.

# ## 2.6 Relationship between most significant feature and prediction
# 
# From the data above and figure below we can see that the most significant factor with abs coefficient around 3.7 is lstat - share of low status population.

# In[77]:


plt.figure(figsize=(10,6), dpi=300)
sns.barplot(prediction.params.index.drop('const'), prediction.params.drop('const'))
plt.xticks(rotation=45)
plt.title('Coefficients of features', size=15)
plt.xlabel("Feature", size=12)
plt.ylabel("Coeffecient", size=12);


# In[78]:


plt.figure(figsize=(8, 6), dpi=300)
sns.scatterplot(target, data.iloc[:,:-1]['lstat'], color='purple')
# sns.histplot(resids, color='purple')
plt.title('Relationship between house value and share of low status population',
          size=15)
plt.xlabel("House value", size=12)
plt.ylabel("Share of poor people", size=12);


# We can see that correlation is strong but not quite linear. As share of low status pop is low, other factors more significantly impact on house value.

# ## 3.1 Improve model interpretability  
# In order to improve interpretability of model we will try 2 things:
# 
# - drop 50k-price houses;
# - drop features with high collinearity - VIF over 5

# In[79]:


new_target = target[target < 50]
new_features = data.iloc[:,:-1][target < 50]

means = new_features.mean(axis=0)
stds = new_features.std(axis=0)

new_features = (new_features - means) / stds

X = sm.add_constant(new_features)
model = sm.OLS(new_target, X)
prediction = model.fit()
print(prediction.summary())


# We can see that R-squared of model has improved and lstat has became less influential.
# 
# 
# Lets try to reduce collinearity.

# In[80]:


calculate_vif(X)


# In[81]:


calculate_vif(X.drop(columns=["tax"]))


# Drop of tax feature - the most correlated with other features - puts all VIFs below 5, which is considered good enough.

# In[82]:


model_updated = sm.OLS(new_target, X.drop(columns=["tax"]))
prediction_updated = model_updated.fit()

print(prediction_updated.summary())


# In[83]:


target_pred = prediction_updated.get_prediction(X.drop(
    columns=["tax"])).predicted_mean
slope, intercept = np.polyfit(new_target, target_pred, 1)
best_line = slope * new_target + intercept
plt.figure(figsize=(8,8), dpi=300)
sns.scatterplot(new_target, target_pred, color='purple')
plt.plot(new_target, best_line, color='green')
plt.title('Relationship between target and prediction', size=15)
plt.xlabel('Target', size=12)
plt.ylabel('Prediction', size=12)


# In[84]:


influence = prediction_updated.get_influence()
cooks = influence.cooks_distance
print((cooks[1] < 0.05).sum())
print(sorted(cooks[1])[:10])


# In[85]:


resids = new_target - target_pred
plt.figure(figsize=(8,6), dpi=300)
sns.scatterplot(target_pred, resids, color='purple')
plt.title('Relationship between prediction and residuals', size=15)
plt.xlabel("Prediction", size=12)
plt.ylabel("Residual", size=12);


# In[87]:


plt.figure(figsize=(8,6), dpi=300)
sns.histplot(resids, color='purple')
plt.title('Distribution of residuals', size=15)
plt.xlabel("Residual", size=12)
plt.ylabel("Count", size=12)


# In[88]:


sm.qqplot(resids, fit=True, line ='45', markerfacecolor='purple', alpha=0.2)
plt.title('Distribution of residuals', size=15)


# Well, not sure if assumptions are met better now. At least, not worser :)  
# Let's look at coefficients of features again.

# In[89]:


params = prediction_updated.params
params.sort_values(ascending=False,key=abs)


# Here are coefficients of different features sorted by abs value. Const - intercept.

# There are three the most significant features - dis, rm and lstat, closely followed by quite significant ptratio and nox.
# So, our recommendations for entrepreneur are:
# 
# - **build as close as possible to the employment centers of the city (dis)**;
# 
# - **build in the towns with larger houses with higher average room count (rm)**;
# 
# - **avoid towns with high share of low status population. The lower, the beter. Or at least lower than 5% (lstat)**;
# 
# - build in towns with lower pupil-teacher ration, preferably lower than 17 (ptratio);
# 
# - build in areas with better air quality, with lower level of nitric oxide related to fuel combustion and car traffic (nox)
# 
# (В общем, стройте в элитном поселке на острове/в лесу близко к центру Москвы (Бостона), там где много деревьев и мало граждан.)
