from flask import Flask, request, render_template
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load your trained Decision Tree Classifier model
dt_model = pickle.load(open('decision_model.pkl', 'rb'))

# Load your trained Random Random Classifier model
rf_model = pickle.load(open('random_model.pkl', 'rb'))

# Load your trained Graient Boosting Classifier model
gb_model = pickle.load(open('gb_model.pkl', 'rb'))

# Load your trained Graient Boosting Classifier model
stack_model = pickle.load(open('stack_model.pkl', 'rb'))


@app.route('/')
def churn_prediction_form():
    return render_template("Churn_Pred.html")

@app.route('/predict', methods=['POST'])
def predict_churn():
    # Get the input features from the form
    age = int(request.form['Age'])
    balance = float(request.form['Balance'])
    estimated_salary = float(request.form['EstimatedSalary'])
    num_of_products = int(request.form['NumOfProducts'])
    tenure = int(request.form['Tenure'])
    credit_score = int(request.form['CreditScore'])
    Review = request.form['Review']
        
    # Create a numpy array with the input features
    input_features = np.array([age, balance, estimated_salary, num_of_products, tenure, credit_score]).reshape(1, -1)
    input_text = np.array([Review])
    # Predict the probability of churn using the loaded model
    churn_probability1 = stack_model.predict_proba(input_features)[:, 1][0]
    

    
    return render_template('Churn_Pred.html', pred='Probability of a Customer Churn is: {:.2f}'.format(churn_probability1))
   
if __name__ == '__main__':
    app.run(debug=True)
