#import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as mat

#reading in the data

inputFile = "diabetes.csv"
df = pd.read_csv(inputFile)
df.head()

#cleaning up data

df = df[df.Glucose != 0]
df = df[df.BloodPressure != 0]
df = df[df.SkinThickness != 0]
df = df[df.Insulin != 0]
df = df[df.BMI != 0]

#overview of the data
pd.set_option('display.max_columns', None)
print(df.describe())

#distributions
#skip pregnancies - the distribution isn't that helpful
sns.displot(data = df, x = "Glucose")
sns.displot(data = df, x = "BloodPressure")
sns.displot(data = df, x = "SkinThickness")
sns.displot(data = df, x = "Insulin")
sns.displot(data = df, x = "BMI")
sns.displot(data = df, x = "Age")
#determining correlations

pd.reset_option('display.max_columns')
print(df.corr())

#correlation visualization

sns.heatmap(df.corr(), annot=True, annot_kws={'size': 15}, vmax=0.7)

#pregnancies versus having diabetes

sns.set_theme(style="whitegrid")
sns.countplot(data=df, x="Pregnancies", hue="Outcome")

#age versus having diabetes
#more helpful to have ages in certain categories
age_bins=pd.cut(df["Age"],bins=[20,30,50,70,90],labels=["20-30","30-50","50-70","70-90"])
sns.countplot(data=df, x=age_bins, hue="Outcome")

#blood pressure risk versus having diabetes
bp_bins=pd.cut(df["BloodPressure"],bins=[0,80,90,200],labels=["<80 (normal)","80-89 (at risk)", ">90 (hypertension)"])
sns.countplot(data=df, x=bp_bins, hue="Outcome")

#bmi category versus having diabetes
bmi_bins=pd.cut(df["BMI"], bins = [0,18.5,25,30,100], labels=["<18.5 (underweight)","18.5-25 (normal)","25-30 (overweight)",">30 (obese)"])
sns.countplot(data=df, x=bmi_bins, hue="Outcome")

#more libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#attempting a logistic regression
#yes i am using a tutorial
scaler=StandardScaler()

#not quite sure what the pedigree function is, and outcome is excluded for obvious reasons
x = df.drop(columns=["Outcome","DiabetesPedigreeFunction"])

y = df["Outcome"]

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
regression = LogisticRegression()
regression.fit(x_train, y_train)

y_predictions = regression.predict(x_test)
cm = confusion_matrix(y_test,y_predictions)
print("\n", cm)
sns.heatmap(cm, annot = True, yticklabels = ['No','Yes'], xticklabels = ['No','Yes'])

print(metrics.accuracy_score(y_test, y_predictions))

