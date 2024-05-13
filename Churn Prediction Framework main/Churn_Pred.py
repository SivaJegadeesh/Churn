#Load the dataset in a Dataframe
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_excel("Churn_Prediction_Dataset.xlsx")

# Extracting the selected features
selected_features = ['CustomerId','Surname','Age', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure', 'CreditScore', 'Exited', 'Review', 'Churn Label']
df_selected = df[selected_features].copy()

# Display the first few rows of the DataFrame after scaling
print(df_selected.head())

# Define the feature matrix (X) and target vector (y) using integer index positions
X = df_selected.iloc[:, 2:8]  # Select rows from index 1 onwards, and columns from index 2 to 7 (inclusive)
y = df_selected['Exited']    # Select rows from index 1 onwards, and column at index 8 (Churn Label)

# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

probabilities_train = decision_tree.predict_proba(X_train)[:, 1]
probabilities_train_series = pd.Series(probabilities_train, index=X_train.index)
print(probabilities_train_series)

# Train Random Forest model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

probabilities_train_rf = random_forest.predict_proba(X_train)[:, 1]
probabilities_train_series_rf = pd.Series(probabilities_train_rf, index=X_train.index)
print(probabilities_train_series_rf)
print(len(X_train))  

# Train Gradient Boosting model
gradient_boosting = GradientBoostingClassifier(random_state=42)
gradient_boosting.fit(X_train, y_train)

probabilities_train_gb = gradient_boosting.predict_proba(X_train)[:, 1]
probabilities_train_series_gb = pd.Series(probabilities_train_gb, index=X_train.index)
print(probabilities_train_series_gb)
print(len(X_train))  # Check the length of X_train

# Define the meta-model (Logistic Regression) for stacking
meta_model = LogisticRegression()

# Create the stacking classifier with base models and meta-model
stacking_classifier = StackingClassifier(
    estimators=[('decision_tree', decision_tree), ('random_forest', random_forest), ('gradient_boosting', gradient_boosting)],
    final_estimator=meta_model,
    cv=5  # Number of folds for cross-validation
)

# Train the stacking classifier
stacking_classifier.fit(X_train, y_train)

probabilities_train_stack = stacking_classifier.predict_proba(X_train)[:, 1]
probabilities_train_series_stack= pd.Series(probabilities_train_stack, index=X_train.index)
print(probabilities_train_series_stack)
print(len(X_train))  # Check the length of X_train

# Save the trained model to a file
pickle.dump(decision_tree, open('decision_model.pkl', 'wb'))

# Save the trained model to a file
pickle.dump(random_forest, open('random_model.pkl', 'wb'))

# Save the trained model to a file
pickle.dump(gradient_boosting, open('gb_model.pkl', 'wb'))

# Save the trained model to a file
pickle.dump(stacking_classifier, open('stack_model.pkl', 'wb'))


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



X_RNN = df_selected['Review']  # Select rows from index 1 onwards, and columns from index 2 to 7 (inclusive)
y_RNN = df_selected['Churn Label']    # Select rows from index 1 onwards, and column at index 8 (Churn Label)

# Split the data into training and test sets
XRNN_train, XRNN_test, yRNN_train, yRNN_test = train_test_split(X_RNN,y_RNN, test_size=0.25, random_state=42)

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(XRNN_train)
XRNN_train_sequences = tokenizer.texts_to_sequences(XRNN_train)
XRNN_test_sequences = tokenizer.texts_to_sequences(XRNN_test)
max_length = 100  # Define maximum sequence length
X_train_pad = pad_sequences(XRNN_train_sequences, maxlen=max_length)
X_test_pad = pad_sequences(XRNN_test_sequences, maxlen=max_length)

# Define the RNN architecture
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
RNN_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=64),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
RNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the RNN model
RNN_model.fit(X_train_pad, y_train, epochs=20, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate the model
y_pred = (RNN_model.predict(X_test_pad) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Plot the pie chart using the probabilities calculated using the churn label

churn_probabilities_train= RNN_model.predict(X_train_pad).flatten()
churn_labels_predicted_train = (churn_probabilities_train > 0.5).astype(int)


# Calculate percentage of churn probabilities
low_churn = np.sum((churn_probabilities_train >= 0) & (churn_probabilities_train < 0.35))
medium_churn = np.sum((churn_probabilities_train >= 0.35) & (churn_probabilities_train < 0.75))
high_churn = np.sum(churn_probabilities_train >= 0.75)

print(low_churn)
print(high_churn)
print(medium_churn)

with open('RNN_model1.pkl', 'wb') as f:
    pickle.dump(RNN_model, f)

    
from keras.models import load_model


# Save the RNN model
RNN_model.save('RNN_model1.h5')

# Load the RNN model
RNN_model = load_model('RNN_model1.h5')





