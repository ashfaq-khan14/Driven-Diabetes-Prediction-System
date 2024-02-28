# diabaties-prediction
<h2 align="left">Hi 👋! Mohd Ashfaq here, a Data Scientist passionate about transforming data into impactful solutions. I've pioneered Gesture Recognition for seamless human-computer interaction and crafted Recommendation Systems for social media platforms. Committed to building products that contribute to societal welfare. Let's innovate with data! 





</h2>

###


<img align="right" height="150" src="https://i.imgflip.com/65efzo.gif"  />

###

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" height="30" alt="javascript logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/typescript/typescript-original.svg" height="30" alt="typescript logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" height="30" alt="react logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" height="30" alt="html5 logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" height="30" alt="css3 logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="30" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/csharp/csharp-original.svg" height="30" alt="csharp logo"  />
</div>

###

<div align="left">
  <a href="[Your YouTube Link]">
    <img src="https://img.shields.io/static/v1?message=Youtube&logo=youtube&label=&color=FF0000&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="youtube logo"  />
  </a>
  <a href="[Your Instagram Link]">
    <img src="https://img.shields.io/static/v1?message=Instagram&logo=instagram&label=&color=E4405F&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="instagram logo"  />
  </a>
  <a href="[Your Twitch Link]">
    <img src="https://img.shields.io/static/v1?message=Twitch&logo=twitch&label=&color=9146FF&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="twitch logo"  />
  </a>
  <a href="[Your Discord Link]">
    <img src="https://img.shields.io/static/v1?message=Discord&logo=discord&label=&color=7289DA&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="discord logo"  />
  </a>
  <a href="[Your Gmail Link]">
    <img src="https://img.shields.io/static/v1?message=Gmail&logo=gmail&label=&color=D14836&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="gmail logo"  />
  </a>
  <a href="[Your LinkedIn Link]">
    <img src="https://img.shields.io/static/v1?message=LinkedIn&logo=linkedin&label=&color=0077B5&logoColor=white&labelColor=&style=for-the-badge" height="35" alt="linkedin logo"  />
  </a>
</div>

###



<br clear="both">


###


---

# Diabetic Prediction
  ![WhatsApp Image 2024-02-28 at 21 39 01_3683f641](https://github.com/ashfaq-khan14/diabaties-prediction/assets/120010803/460ac1a7-884e-4e94-97fa-d138f4e24145)


## Overview
This project focuses on predicting the likelihood of a person developing diabetes based on various health indicators. By analyzing features such as glucose levels, blood pressure, and body mass index (BMI), the model can accurately classify individuals as either diabetic or non-diabetic, enabling early detection and intervention for at-risk individuals.

## Dataset
The project utilizes a dataset containing health information of patients, including demographic details, medical history, and diagnostic measurements. Each sample in the dataset is labeled as either diabetic or non-diabetic based on medical diagnosis.

## Features
- *Pregnancies*: Number of times pregnant.
- *Glucose*: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
- *Blood Pressure*: Diastolic blood pressure (mm Hg).
- *Skin Thickness*: Triceps skinfold thickness (mm).
- *Insulin*: 2-Hour serum insulin (mu U/ml).
- *BMI*: Body mass index (weight in kg/(height in m)^2).
- *Diabetes Pedigree Function*: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history).
- *Age*: Age of the individual.
- *Outcome*: Target variable, indicating whether the individual is diabetic (1) or non-diabetic (0).

## Models Used
- *Logistic Regression*: Simple and interpretable baseline model.
- *Random Forest*: Ensemble method for improved predictive performance.
- *Gradient Boosting*: Boosting algorithm for enhanced accuracy and efficiency.

## Evaluation Metrics
- *Accuracy*: Measures the proportion of correctly classified samples.
- *Precision*: Measures the proportion of true positive predictions among all positive predictions.
- *Recall*: Measures the proportion of true positive predictions among all actual positive samples.
- *F1 Score*: Harmonic mean of precision and recall, providing a balance between the two metrics.
- *Area Under the ROC Curve (AUC-ROC)*: Measures the ability of the model to distinguish between classes.

## Installation
1. Clone the repository:
   
   git clone https://github.com/ashfaq-khan14/diabaties-prediction/edit/main/README.md
   
2. Install dependencies:
   
   pip install -r requirements.txt
   

## Usage
1. Preprocess the dataset (if necessary) and prepare the features and target variable.
2. Split the data into training and testing sets.
3. Train the classification models using the training data.
4. Evaluate the models using the testing data and appropriate evaluation metrics.
5. Make predictions on new data using the trained models.

## Example Code
python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('diabetes_data.csv')

# Split features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))


## Future Improvements
- *Feature Engineering*: Explore additional features or transformations to improve model accuracy.
- *Imbalanced Data Handling*: Implement techniques to handle imbalanced data and improve model performance on detecting diabetic cases.
- *Model Ensembling*: Combine predictions from multiple models for improved accuracy.
- *Deployment*: Deploy the trained model as a predictive tool for healthcare professionals to identify individuals at risk of diabetes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

