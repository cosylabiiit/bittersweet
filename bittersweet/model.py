import pickle
from sklearn.ensemble.forest import RandomForestClassifier

class Model():
    def __init__(self, bitter_model_path, bitter_features_path, sweet_model_path, sweet_features_path):
        self.bitter_model = pickle.load(open(bitter_model_path, 'rb'))
        self.bitter_features = pickle.load(open(bitter_features_path, 'rb'))
        self.sweet_model = pickle.load(open(sweet_model_path, 'rb'))
        self.sweet_features = pickle.load(open(sweet_features_path, 'rb'))

    def predict_bitter(self, data):
        # Subset to relevant columns
        d = data[self.bitter_features]

        # Predict
        bitter_prob = self.bitter_model.predict_proba(d)
        bitter_taste = self.bitter_model.predict(d)
        
        return bitter_prob, bitter_taste
        
    def predict_sweet(self, data):
        # Subset to relevant columns
        d = data[self.sweet_features]

        # Predict
        sweet_prob = self.sweet_model.predict_proba(d)
        sweet_taste = self.sweet_model.predict(d)

        return sweet_prob, sweet_taste
        
    def predict(self, data):
        bitter_prob, bitter_taste = self.predict_bitter(data)
        sweet_prob, sweet_taste = self.predict_sweet(data)
        
        return bitter_taste, bitter_prob, sweet_taste, sweet_prob

    
