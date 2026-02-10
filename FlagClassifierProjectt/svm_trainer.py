from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_svm(X_train, y_train, X_val, y_val):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_train_enc)

    preds = svm.predict(X_val)
    print("Accuracy:", accuracy_score(y_val_enc, preds))
    print(classification_report(y_val_enc, preds, target_names=le.classes_))
    
    return svm, le
