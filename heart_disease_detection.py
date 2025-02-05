#!/usr/bin/env python
# coding: utf-8

# ### 2.Heart Disease Detection

# In[64]:


import pandas as pd
df = pd.read_csv("heart.csv")
df.head()


# In[65]:


x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[66]:


#Naive Bayes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
model=GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

# Import the necessary libraries
from sklearn.metrics import accuracy_score

# Calculate accuracy using the accuracy_score function
nb_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using accuracy_score: {nb_accuracy}")


# In[67]:


#Decision Tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
dt_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree Classifier: {dt_accuracy}")


# In[68]:


#SVM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Train the Decision Tree Classifier
model = SVC(kernel = 'linear', random_state = 0)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
svm_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM: {svm_accuracy}")


# In[74]:


#KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Train the Decision Tree Classifier
model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2 )  
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
knn_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN: {knn_accuracy}")


# In[70]:


#Random Forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestClassifier instead of RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)  # Use the classifier
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {rf_accuracy}")


# In[71]:


#Linear Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

result_lr = accuracy_score(y_test, y_pred)
print("Accuracy:", result_lr)


# In[72]:


y_plot=[nb_accuracy,dt_accuracy,rf_accuracy,svm_accuracy,knn_accuracy,result_lr]
X_plot=['NB','DT','RF','SVM','KNN','LR']


# In[75]:


plt.title('HEART DISEASE')
plt.plot(X_plot, y_plot, 'o', color='blue')  # 'o' denotes points
plt.plot(X_plot, y_plot)
plt.grid(True)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




