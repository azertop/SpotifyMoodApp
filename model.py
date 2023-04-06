import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

# On charge les données à partir du csvimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

# On charge les données à partir du csv
df = pd.read_csv("data_moods.csv")

# On sélectionne les features pertinentes

features=["danceability","acousticness","energy","instrumentalness","liveness","valence","loudness","speechiness","key","tempo"]
X = df[features]
y = df['mood']

#Afin de passer de str à int à l'aide d'un Encoder
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# On utilise un Pipeline afin de standardiser les données puis d'utiliser un SVM
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='linear'))
])

# On entraine le modèle
pipe.fit(X_train, y_train)

# On teste le modèle
y_pred = pipe.predict(X_test)

# On évalue le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle : {accuracy}")

with open('model.pickle', 'wb') as f:
    pickle.dump(pipe, f)
    
with open('labelencoder.pickle', 'wb') as f:
    pickle.dump(le, f)

