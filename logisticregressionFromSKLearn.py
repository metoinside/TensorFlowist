from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


# load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# split data into test and training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# normalize the samples
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

print lr.predict_proba(X_test_std[0,:])