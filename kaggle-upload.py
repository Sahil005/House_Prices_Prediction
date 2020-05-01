#!/usr/bin/env python
# coding: utf-8

# # TRAIN DATA CLEANUP

# In[ ]:


import pandas as pd 
df = pd.read_csv("C:/Users/User/Downloads/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("C:/Users/User/Downloads/house-prices-advanced-regression-techniques/test.csv")
pd.options.display.max_columns = 4000
pd.options.display.max_rows = 4000


# In[ ]:


#df.head()
test.head()


# In[ ]:


import numpy as np;
import matplotlib.pyplot as plt

import seaborn as sns;
fig_dims = (20,16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(df.isnull(),cmap='viridis')


# In[ ]:


len(df)
df.isnull().sum()


# In[ ]:


df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis =1, inplace = True)
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])


# # TEST DATA CLEANUP

# In[ ]:


import numpy as np;
import matplotlib.pyplot as plt

import seaborn as sns;
fig_dims = (20,16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(df.isnull(),cmap='viridis')


# In[ ]:


test.isnull().sum()


# In[ ]:


test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis =1, inplace = True)


# In[ ]:


test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mean())
test['MasVnrType']=test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtExposure']=test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['FireplaceQu']=test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])
test['GarageYrBlt']=test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0])
test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test['Electrical']=test['Electrical'].fillna(test['Electrical'].mode()[0])
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])


# In[ ]:


test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0])
test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0])
test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])
test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])
test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mean())
test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mode()[0])
test['SaleType']=test['SaleType'].fillna(test['SaleType'].mode()[0])
test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])
test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])


# # Getting DUMMY VARIABLES

# In[ ]:


df.info()


# In[ ]:


test.info()


# In[ ]:


Combined_df = pd.concat([df,test],axis = 0)


# In[ ]:


Combined_df.shape


# In[ ]:


for_object = Combined_df.select_dtypes(include='object')
columns =list(for_object.columns)
com_with_dummy = pd.get_dummies(Combined_df,columns,drop_first = True)
com_with_dummy.shape


# In[ ]:


com_with_dummy = com_with_dummy.loc[:,~com_with_dummy.columns.duplicated()]


# In[ ]:


train_df = com_with_dummy.iloc[:1460,:]


# In[ ]:


train_df.columns


# In[ ]:


test_df = com_with_dummy.iloc[1460:,:]


# In[ ]:


test_df.shape


# In[ ]:


test_df = test_df[train_df.columns]


# In[ ]:


test_df.drop(['SalePrice'],axis = 1, inplace = True)


# In[ ]:


print(test_df)


# In[ ]:


import xgboost
from xgboost import XGBRegressor


# In[ ]:


X_train = train_df.drop(['SalePrice'],axis =1)
y_train = train_df['SalePrice']


# In[ ]:


Classifier = xgboost.XGBRegressor()


# In[ ]:


print(test_df1)


# In[ ]:


Classifier.fit(X_train,y_train)


# In[ ]:


y_pred = Classifier.predict(test_df)


# In[ ]:


y_pred


# In[ ]:


pred = pd.DataFrame(y_pred)


# In[ ]:


sub_df = pd.read_csv('C:/Users/User/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_df1 = pd.DataFrame(sub_df)
sub_df2 = sub_df1['SalePrice']
#print(sub_df2)


# In[ ]:


test_df = pd.concat([test_df,sub_df2],axis =1)
y_test = test_df['SalePrice']
print(y_test)


# In[ ]:


y_test = y_test.to_numpy()
y_test = y_test.reshape(-1,1)
type(y_test)


# In[ ]:


new_df = pd.concat([sub_df,pred],axis = 1)


# In[ ]:


new_df.to_csv('our_file.csv',index=False)

