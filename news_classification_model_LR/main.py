from src.data_preprocessing import preprocessing_data
from src.train import model_training
from src.evaluate import evaluate

def run_pipeline():
    preprocessing_data()
    model_training() 
    print("next up: evaluation")
    evaluate()


if __name__=="__main__":
    run_pipeline()
    