import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
df = pd.read_csv("/content/tushali/Data/Iris.csv")
features =['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
target = ['Species']
X_train,X_test,y_train,y_test = train_test_split(df[features],df[target], test_size=0.2,shuffle =True)

clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy of classifier is:  ",{accuracy_score(y_test,y_pred)*100})