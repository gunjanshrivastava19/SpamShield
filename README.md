**SpamShield Project**
This project is a machine learning-based Spam Detection System that classifies SMS messages as Spam or Not Spam (Ham). It uses text preprocessing, TF-IDF feature extraction, and a Logistic Regression model to make predictions. The system also includes a keyword-based check for detecting obvious spam messages.

 **AI-Powered SMS Spam Detection System**
SpamShield is a machine-learning-based SMS spam detection application that classifies messages as **SPAM** or **NOT SPAM**.
The project combines **Natural Language Processing (NLP), TF-IDF feature extraction, Logistic Regression, and rule-based keyword detection** with an interactive **Streamlit** interface.
**Features**

-  Real-time SMS spam detection
-  Machine Learning classification using Logistic Regression
-  TF-IDF feature extraction with unigrams and bigrams
-  Rule-based detection for high-signal spam keywords
-  Interactive dataset dashboard
-  Accuracy, Precision, Recall and F1 Score
-  Confusion Matrix visualization
-  Prediction confidence score
-  Explanation of how each prediction was generated
-  Interactive Streamlit web interface
-  Cached model training for faster interaction

   **How SpamShield Works**

            SMS Message
                 ↓
          Text Preprocessing
                 ↓
        TF-IDF Feature Extraction
                 ↓
         Logistic Regression
                 ↓
      Spam Probability Calculation
                 ↓
       Strong Keyword Detection
                 ↓
          Final Classification
                 ↓
        SPAM / NOT SPAM
   
 **SpamShield uses a hybrid detection approach**
The Logistic Regression model analyzes the TF-IDF representation of the message, while a rule-based detector checks for high-signal spam keywords such as:
**lottery, prize, winner, claim, free, cash, reward, credited, won**
The final prediction uses the original 35% spam-probability threshold together with the keyword-based detection logic.

**Technology Stack**

| Technology          | Purpose                            |
| ------------------- | ---------------------------------- |
| Python              | Core programming language          |
| Pandas              | Dataset handling and preprocessing |
| Scikit-learn        | Machine Learning                   |
| TF-IDF              | Text feature extraction            |
| Logistic Regression | Spam classification                |
| Streamlit           | Interactive web interface          |
| Regular Expressions | Text cleaning                      |

**Dataset**
SpamShield uses the SMS Spam Collection dataset containing:
**5,574 total messages
4,825 Ham messages
749 Spam messages
Approximately 13.44% spam messages**
The dataset contains the original v1 and v2 columns, representing the message label and SMS text.

**Model Performance**
The model is evaluated using a stratified 80/20 train-test split with random_state=42.
| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **88.79%** |
| Precision | **54.86%** |
| Recall    | **94.00%** |
| F1 Score  | **69.29%** |

Confusion Matrix
                 Predicted
               Ham     Spam

Actual Ham     849     116
Actual Spam      9     141
The model achieves a 94% recall for spam messages, meaning it identifies the majority of spam messages in the test set.
Performance metrics are calculated dynamically from the project's test split and are not hard-coded.

**Application**
SpamShield provides an interactive Streamlit interface with several sections:

 **Detect Message**
Enter an SMS message and receive:

**SPAM / NOT SPAM prediction
Confidence score
Spam probability
Detection explanation**

📊**Dashboard**
Provides an overview of:
**Total messages
Spam messages
Ham messages
Spam percentage
Spam vs Ham distribution**

🤖**Model Information**
Displays:
**Machine Learning algorithm
TF-IDF configuration
Train/Test split
Spam threshold
Model performance
Confusion matrix**

⚙️**How It Works**
Explains the complete SpamShield detection pipeline.

ℹ️**About Project**
Provides an overview of the project's purpose and technologies.


📂**Project Structure**
SpamShield
─ app.py
─ spam_model.py
─ SpamDetectionModel_legacy.py
─ spam.csv
─ requirements.txt
─ .gitignore
─ README.md

**File Description**

**app.py**
Main Streamlit application and user interface.

**spam_model.py**
Reusable Machine Learning backend containing preprocessing, training, evaluation and prediction logic.

**SpamDetectionModel_legacy.py**
Original standalone implementation retained to document the project's earlier development stage.

**spam.csv**
SMS Spam Collection dataset.

**requirements.txt**
Python dependencies required to run the application.



**Code Explanation**
This project builds a Spam Detection System using Machine Learning and a small rule-based filter. Below is the explanation of how the code works:

**1. Importing Libraries**

The required libraries are imported:
1.pandas for handling the dataset
2.re for text cleaning using regular expressions
3.train_test_split for splitting data
4.TfidfVectorizer for converting text into numerical features
5.LogisticRegression for building the classification model

**2. Loading and Preparing the Dataset**

1.The dataset spam.csv is loaded using pandas.
2.Only the relevant columns (v1, v2) are selected.
3.Columns are renamed to label and message.
4.Missing values are removed.
5.Labels are converted into numeric form:
ham → 0
spam → 1
This prepares the data for training the model.

**3. Text Cleaning**

1.A function clean_text() is created to preprocess the messages:
2.Converts text to lowercase.
3.Removes numbers and special characters.
4.Keeps only alphabets and spaces.
5.All messages in the dataset are cleaned using this function.

**4. Splitting the Data**

1.The dataset is split into:
**80% Training Data
20% Testing Data**
2.Stratified splitting ensures equal distribution of spam and ham messages in both sets.

**5. Feature Extraction using TF-IDF**

The TfidfVectorizer converts text messages into numerical vectors.
1.ngram_range=(1,2) means the model considers:
2.Single words (unigrams)
3.Two-word combinations (bigrams)
This helps the model understand important word patterns like “free cash”.

**6. Model Training**
1.A Logistic Regression model is created and trained using the vectorized training data.
2.The model learns patterns that distinguish spam messages from normal messages.

**7. Strong Keyword Detection**
A list of strong spam keywords is defined:
**lottery
prize
winner
claim
free
cash**
If any of these words appear in a message, it is directly classified as spam without using the model.

**8. Real-Time Prediction**

The program runs in a loop and:
1.Takes user input.
2.Stops if the user types exit.
3.Cleans the input message.
4.Checks for strong spam keywords.
5.If no keyword is found:
**Converts the message into TF-IDF format.
Predicts spam probability using the trained model.**
6.If probability > 0.35 → Classified as **SPAM**
Otherwise → **NOT SPAM**
7.Displays the confidence score.

<img width="1089" height="811" alt="image" src="https://github.com/user-attachments/assets/99d57cff-319a-4449-930f-57a443961bd1" />
<img width="1128" height="277" alt="Screenshot 2026-02-25 194027" src="https://github.com/user-attachments/assets/5b3d7b82-a576-49ab-a3d5-699055b0b9cf" />
<img width="1029" height="286" alt="Screenshot 2026-02-25 194645" src="https://github.com/user-attachments/assets/214cc507-94f5-4676-9ba7-fdba968e0e05" />


