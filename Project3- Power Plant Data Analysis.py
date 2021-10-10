#!/usr/bin/env python
# coding: utf-8

# # Imporing Libraries and Dataset

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[8]:


data_train= pd.read_csv(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Power Plant Data Analysis\train.csv", delimiter=",")


# In[9]:


data_train.head()


# In[10]:


data_train.shape


# In[11]:


y_train= data_train[" EP"]


# In[12]:


del data_train[" EP"]


# In[13]:


data_train.head()


# In[14]:


y_train.head()


# # Structure of Dataset

# In[15]:


data_train.describe()


# In[16]:


y_train.shape


# # Checking for Null values

# In[17]:


data_train.isnull().sum()


# In[18]:


data_train.isna().sum()


# In[19]:


y_train.isnull().sum()


# In[20]:


y_train.isna().sum()


# # Exploratory Data Analysis

# In[21]:


# Statistics

min_EP= y_train.min()
max_EP= y_train.max()
mean_EP= y_train.mean()
median_EP= y_train.median()
std_EP= y_train.std()


# In[22]:


# Quartile calculator

first_quar= np.percentile(y_train, 25)
third_quar= np.percentile(y_train, 75)
inter_quar= third_quar - first_quar


# In[23]:


# Print Statistics

print("Statistics for combined cycle Power Plant:\n")
print("Minimum EP:", min_EP)
print("Maximum EP:", max_EP)
print("Mean EP:", mean_EP)
print("Median EP:", median_EP)
print("Standard Deviation of EP:", std_EP)
print("First Quartile of EP:", first_quar)
print("Third Quartile of EP:", third_quar)
print("InterQuartile of EP:",inter_quar)


# # Plotting

# In[24]:


sns.set(rc={"figure.figsize":(5,5)})
sns.distplot(data_train, bins=30, color= "orange")
plt.show()


# # Correlation

# In[25]:


corr_df=data_train.copy()
corr_df["EP"]=y_train
corr_df.head()


# In[26]:


sns.set(style="ticks", color_codes=True)
plt.figure(figsize=(12,12))
sns.heatmap(corr_df.astype("float32").corr(), linewidths=0.1, square=True, annot=True)
plt.show()


# # Features Plot

# In[27]:


# Print all Features

data_train.columns


# In[28]:


plt.plot(corr_df["# T"], corr_df["EP"], "+", color= "green")
plt.plot(np.unique(corr_df["# T"]), np.poly1d(np.polyfit(corr_df["# T"], corr_df["EP"], 1))
        (np.unique(corr_df["# T"])), color="yellow")
plt.xlabel("Temperature", fontsize=12)
plt.ylabel("EP", fontsize=12)
plt.show()


# In[29]:


plt.plot(corr_df[" V"], corr_df["EP"], "o", color= "pink")
plt.plot(np.unique(corr_df[" V"]), np.poly1d(np.polyfit(corr_df[" V"], corr_df["EP"], 1))
        (np.unique(corr_df[" V"])), color="blue")
plt.xlabel("Exhaust Vaccum", fontsize=12)
plt.ylabel("EP", fontsize=12)

plt.show()


# In[30]:


plt.plot(corr_df[" AP"], corr_df["EP"], "o", color= "orange")
plt.plot(np.unique(corr_df[" AP"]), np.poly1d(np.polyfit(corr_df[" AP"], corr_df["EP"], 1))
        (np.unique(corr_df[" AP"])), color="green")
plt.xlabel("Ambient Pressure", fontsize=12)
plt.ylabel("EP", fontsize=12)
plt.show()


# In[31]:


plt.plot(corr_df[" RH"], corr_df["EP"], "o", color= "seagreen")
plt.plot(np.unique(corr_df[" RH"]), np.poly1d(np.polyfit(corr_df[" RH"], corr_df["EP"], 1))
        (np.unique(corr_df[" RH"])), color="blue")
plt.xlabel("Relative Humidity", fontsize=12)
plt.ylabel("EP", fontsize=12)

plt.show()


# In[32]:


fig, ax=plt.subplots(ncols=4, nrows=1, figsize=(20,10))
index=0
ax=ax.flatten()
for i,v in data_train.items():
    sns.boxplot(y=i, data=data_train, ax=ax[index], color= "orangered")
    index+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)


# In[33]:


sns.set(style="whitegrid")
features_plot=data_train.columns

sns.pairplot(data_train[features_plot]);
plt.tight_layout
plt.show()


# # Feature Scaling

# In[34]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit_transform(data_train)


# # Gradient Descent Model

# In[35]:


x_train= data_train


# In[36]:


x_train.shape, y_train.shape


# In[37]:


from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor(learning_rate=1.9, n_estimators=2000)
gbr


# In[38]:


gbr.fit(x_train, y_train)


# In[43]:


x_test= np.genfromtxt(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Power Plant Data Analysis\test.csv", delimiter=",")
y_train.ravel(order="A")
y_pred=gbr.predict(x_test)


# In[44]:


y_pred


# # Model Evaluation

# In[45]:


gbr.score(x_train, y_train)


# # Saving the Prediction

# In[46]:


np.savetxt("Predict_csv", y_pred, fmt="%.5f")


# In[ ]:




