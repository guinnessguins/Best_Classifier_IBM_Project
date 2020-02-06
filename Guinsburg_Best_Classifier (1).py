#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

#  Lets define feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


lb = preprocessing.LabelBinarizer()
y=lb.fit_transform(df['loan_status'])
le_ls = preprocessing.LabelEncoder()
le_ls.fit(['Paidoff','Collection'])
#y = df['loan_status']
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[20]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



# Training - i.e. fitting - and Evaluating the model
from sklearn.neighbors import KNeighborsClassifier
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

from sklearn import metrics

from sklearn.metrics import jaccard_similarity_score as jaccard
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
for n in range(1,Ks):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train.ravel())
    yhat=knn.predict(X_test)
    mean_acc[n-1] = jaccard(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
  

#Jaccard Index
mean_acc


# In[21]:


# Here I plot the Jaccard Index against many k neighbors estimators to see what is the best fit
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[22]:


# It seems the best fit is achieved with k=7 AND I THEREFORE USE 7 AS MY KNN ESTIMATOR

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train.ravel())
yhat=knn.predict(X_test)
yhat
from sklearn import metrics
print("Test set Accuracy Jaccard: ", metrics.accuracy_score(y_test, yhat))
print("Train set Accuracy F1 Score:", f1_score(y_train, knn.predict(X_train)))


# # Decision Tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
t = 20
mean_acc = np.zeros((t-1))
std_acc = np.zeros((t-1))
for n in range(1,t):
    DefaultTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    DefaultTree.fit(X_train,y_train)
    yhat = DefaultTree.predict(X_test)
    mean_acc[n-1] = jaccard(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    # THE BEST FIT WAS ACHIEVED WITH DEPTH=1 OR 2 WITH 1 BEING SLIGHTLY BETTER
# I THEREFORE HAVE CHOSEN 1
# Here I plot the Jaccard Index against many k neighbors estimators to see what is the best fit
plt.plot(range(1,t),mean_acc,'g')
plt.fill_between(range(1,t),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[25]:


# It seems both n=1 and 2 yield the same accuracy as measured by the Jaccard Index
#Using the principle of parsimony I use only n=1 as the optimal choice
DefaultTree = DecisionTreeClassifier(criterion="entropy", max_depth = 1)
DefaultTree.fit(X_train,y_train)
yhat = DefaultTree.predict(X_test)
print("DecisionTrees's Jaccard: ", metrics.accuracy_score(y_test, yhat))
print("DecisionTree's Accuracy-F1 Score:", f1_score(y_test, yhat, average='weighted'))


# # Support Vector Machine 

# In[26]:


lb = preprocessing.LabelBinarizer()
y=lb.fit_transform(df['loan_status'])
from sklearn import svm
from sklearn.metrics import f1_score
# Classifiers can either be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
type=['linear','poly', 'rbf', 'sigmoid']

measures =[]
for i in type:
    clf = svm.SVC(kernel=i)
    clf.fit(X_train, y_train.ravel()) 
    yhat = clf.predict(X_test)
    irow=[i,f1_score(y_test, yhat, average='weighted'), metrics.accuracy_score(y_test, yhat)]
    measures.append(irow)
    
measures
pd.DataFrame(measures,columns=['Kernel','F1 Score', 'Jaccard'])


# In[36]:


# The SVM seems very uncertain as to what is the best parametric choice of Kernel
# I will choose RBF based on the fact that in the SVM exercise Saeed has presented to us, the f1 score is the choice of success metric.
# For the sake of completeness of this exercise and for better visualization of the results, I compute next the confusion matrix associated to the RBF kernel
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train.ravel()) 
yhat = clf.predict(X_test)
irow=[i,f1_score(y_test, yhat, average='weighted'), metrics.accuracy_score(y_test, yhat)]
measures.append(irow)

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

  # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Paidoff','Collection'],normalize= False,  title='Confusion matrix')  


# # Logistic Regression

# In[37]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


# In[38]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train.ravel())
LR


# In[40]:


yhat = LR.predict(X_test)
yhat


# # Model Evaluation using Test set

# In[ ]:





# First, download and load the test set:

# In[41]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[42]:


test_df = pd.read_csv('loan_test.csv')
test_df.dtypes


# In[43]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature2 = test_df[['Principal','terms','age','Gender','weekend']]
Feature2 = pd.concat([Feature2,pd.get_dummies(test_df['education'])], axis=1)
Feature2.drop(['Master or Above'], axis = 1,inplace=True)
X_tst = Feature2
X_tst = preprocessing.StandardScaler().fit(X_tst).transform(X_tst.astype(float))
lb = preprocessing.LabelBinarizer()
y_tst=lb.fit_transform(test_df['loan_status'])
le_ls = preprocessing.LabelEncoder()
le_ls.fit(['Paidoff','Collection'])
y_tst.shape


# In[44]:


yhat_Knn=knn.predict(X_tst)
yhat_SVM=clf.predict(X_tst)
yhat_Logist = LR.predict(X_tst)
yhat_Tree = DefaultTree.predict(X_tst)


# In[47]:


algorithm={'Knn':yhat_Knn, 'SVM':yhat_SVM, 'Logist':yhat_Logist, 'Tree':yhat_Tree}
algorithm2=['Knn', 'Tree', 'SVM', 'Logist' ]
measures= []
col=['Algorithm','F1-Score', 'Jaccard', 'Log Loss']
for i in algorithm2:
    #y_i='yhat_'+i
    if i== 'Logist': 
        irow = [i, f1_score(y_tst, algorithm[i], average='weighted'), metrics.accuracy_score(y_tst, algorithm[i]),log_loss(y_tst, algorithm[i]) ]
        measures.append(irow)
    else:
        irow = [i, f1_score(y_tst, algorithm[i], average='weighted'), metrics.accuracy_score(y_tst, algorithm[i]), 'NA']
        measures.append(irow)
    



measures
pd.DataFrame(measures,columns=col)

#It seems that the SVM with RBF Kernel beats all other classification algorithms.


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
