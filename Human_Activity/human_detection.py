#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
training_data = pd.read_csv(r'C:\Users\DINESH 07\Downloads\train.csv')
testing_data = pd.read_csv(r'C:\Users\DINESH 07\Downloads\test.csv')

# Extract features and labels
y_train = training_data['Activity']
X_train = training_data.drop(columns=['Activity', 'subject'])

y_test = testing_data['Activity']
X_test = testing_data.drop(columns=['Activity', 'subject'])

# Initialize accuracy_scores list
accuracy_scores = [0, 0, 0, 0]

# Support Vector Classifier
clf = SVC().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction) * 100

# Logistic Regression
clf3 = LogisticRegression().fit(X_train, y_train)
prediction = clf3.predict(X_test)
accuracy_scores[1] = accuracy_score(y_test, prediction) * 100

# K Nearest Neighbors
clf1 = KNeighborsClassifier().fit(X_train, y_train)
prediction1 = clf1.predict(X_test)
accuracy_scores[2] = accuracy_score(y_test, prediction1) * 100  # Corrected index

# Random Forest
clf2 = RandomForestClassifier().fit(X_train, y_train)
prediction2 = clf2.predict(X_test)
accuracy_scores[3] = accuracy_score(y_test, prediction2) * 100  # Corrected index

# Additional analysis
df = training_data.isnull().values.any()
df1 = testing_data.isnull().values.any()
count_of_each_activity = y_train.value_counts()

# Create DataFrame
final_list = []
small_list = [df1, df, count_of_each_activity.index.tolist()] + accuracy_scores
final_list.append(small_list)
Final_output = pd.DataFrame(final_list, columns=['count_null_values_test', 'count_null_values_train', 'activities', 'Support Vector Classifier accuracy', 'Logistic Regression accuracy', 'K Nearest Neighbors Classifier accuracy', 'Random Forest Classifier accuracy'])

# Display pie chart
plt.rcParams.update({'figure.figsize': [20, 20], 'font.size': 24})
plt.pie(count_of_each_activity, labels=count_of_each_activity.index, autopct='%0.2f')
plt.show()

# Display the final output DataFrame
print(Final_output)


# In[27]:


file_name= r'C:\Users\DINESH 07\Downloads\aml project.xlsx'
Final_output.to_excel(file_name)


# In[29]:


Final_output


# In[30]:


training_data.shape


# In[31]:


testing_data.shape


# In[32]:


classification=classification_report(y_test, prediction2)
print(classification)


# In[33]:


classification1=classification_report(y_test, prediction1)
print(classification1)


# In[ ]:





# In[34]:


import matplotlib.pyplot as plt


# In[35]:


algo=["Randomforest","KNN"]
acc=[0.93,0.90]
act=["laying","sitting","standing","walking","walking_down","walking_up"]
pre1=[1,0.91,0.83,0.85,0.94,0.90]
pre2=[1,0.91,0.90,0.90,0.96,0.89]
plt.plot(algo,acc)


# In[100]:


plt.plot(act,pre1,label="RANDOM_FOREST")
plt.plot(act,pre2,label="KNN")
plt.legend()


# In[ ]:




