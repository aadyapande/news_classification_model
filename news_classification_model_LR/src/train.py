import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.feature_engineering import extract_features
from src.config import MODEL_PATH


def model_training():
    X,y,vectorizer= extract_features()

    print("Splitting training and testing data...")
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

    classifier= LogisticRegression(max_iter=1000)
    print("Training the Logistic Regression Model...")
    classifier.fit(X_train,y_train)

    print("Saving the trained model...")
    with open(MODEL_PATH,"wb") as f:
        pickle.dump((classifier,vectorizer,X_test,y_test),f)

    print("Model is trained and saved")


if __name__=="__main__":
    model_training()
