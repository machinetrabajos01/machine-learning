# Machine Learning Lab

## Project Description

Machine Learning Lab is an educational web application developed with Python and Flask to explain fundamental concepts of Machine Learning and demonstrate a practical Linear Regression application.

The project presents Machine Learning concepts, types of Machine Learning, four different use cases, and a complete Housing Price Prediction example using Linear Regression.

## Technologies

- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Bootstrap 5
- HTML5
- CSS3
- Git
- GitHub

## Machine Learning Topics

The application covers:

- Machine Learning Concepts
- Types of Machine Learning
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Four Machine Learning Use Cases
- Linear Regression Concepts
- Linear Regression Application

## Linear Regression Application

The practical application focuses on Housing Price Prediction.

### Variables

- Independent variable: `House_Area`
- Unit: square meters (m²)
- Dependent variable: `House_Price`
- Unit: Colombian pesos (COP)

The dataset contains 1,000 synthetic records created for academic purposes.

The Linear Regression model is implemented using Scikit-learn.

## Project Structure

```text
machine-learning/
│
├── app.py
├── Procfile
├── requirements.txt
├── README.md
│
├── data/
│   └── housing_dataset.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── housing_regression.png
│
└── templates/
    ├── base.html
    ├── home.html
    ├── ml_concepts.html
    ├── ml_types.html
    ├── use_case_1.html
    ├── use_case_2.html
    ├── use_case_3.html
    ├── use_case_4.html
    ├── regression_concepts.html
    └── regression_application.html