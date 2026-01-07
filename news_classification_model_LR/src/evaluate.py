print("Entering Evaluation")
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from src.config import MODEL_PATH, METRICS_PATH

def evaluate():
    print("Starting Evaluation")
    with open(MODEL_PATH, "rb") as f:
        classifier,vectorizer,X_test,y_test= pickle.load(f)
    
    y_predict=  classifier.predict(X_test)
    accuracy= accuracy_score(y_test,y_predict)

    with open(METRICS_PATH,"w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(y_test,y_predict)}")

    print(f"Model Accuracy:\t {accuracy}")


if __name__=="__main__":
    evaluate()