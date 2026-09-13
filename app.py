from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score,
    f1_score
)


app = Flask(__name__)


# Activity 1

data = pd.read_csv("data/housing_dataset.csv")

print("Dataset loaded successfully")
print("Number of records:", len(data))
print(data.head())

X = data[["House_Area"]]
y = data["House_Price"]

linear_model = LinearRegression()
linear_model.fit(X, y)

predictions = linear_model.predict(X)

r2 = r2_score(y, predictions)

coefficient = linear_model.coef_[0]
intercept = linear_model.intercept_

print("Linear Regression model trained successfully")
print("R² Score:", r2)
print("Coefficient:", coefficient)
print("Intercept:", intercept)


plt.figure(figsize=(10, 6))

plt.scatter(
    data["House_Area"],
    data["House_Price"],
    alpha=0.6
)

plt.plot(
    data["House_Area"],
    linear_model.predict(X)
)

plt.title("House Area vs House Price")
plt.xlabel("House Area (m²)")
plt.ylabel("House Price (COP)")

plt.grid(True)
plt.tight_layout()

plt.savefig("static/housing_regression.png")
plt.close()

print("Regression plot created successfully")


# Activity 2

classification_data = pd.read_csv(
    "data/housing_classification.csv"
)

print("Classification dataset loaded successfully")
print("Classification records:", len(classification_data))
print(classification_data.head())

X_logistic = classification_data[["House_Area"]]
y_logistic = classification_data["Price_Category"]

# Gradient Boosting data

X_gradient = classification_data[
    [
        "House_Area",
        "Bedrooms",
        "Bathrooms",
        "House_Age"
    ]
]

y_gradient = classification_data["Price_Category"]

print("Gradient Boosting variables prepared successfully")
print("Input variables:", list(X_gradient.columns))

# Gradient Boosting training split

X_train_gradient, X_test_gradient, y_train_gradient, y_test_gradient = train_test_split(
    X_gradient,
    y_gradient,
    test_size=0.20,
    random_state=42,
    stratify=y_gradient
)

print("Gradient Boosting data split successfully")
print("Training records:", len(X_train_gradient))
print("Testing records:", len(X_test_gradient))

# Gradient Boosting model training

gradient_model = GradientBoostingClassifier(
    random_state=42
)

gradient_model.fit(
    X_train_gradient,
    y_train_gradient
)

gradient_predictions = gradient_model.predict(
    X_test_gradient
)

print("Gradient Boosting Classifier trained successfully")

# Gradient Boosting evaluation

gradient_accuracy = accuracy_score(
    y_test_gradient,
    gradient_predictions
)

gradient_precision = precision_score(
    y_test_gradient,
    gradient_predictions,
    zero_division=0
)

gradient_recall = recall_score(
    y_test_gradient,
    gradient_predictions,
    zero_division=0
)

gradient_f1 = f1_score(
    y_test_gradient,
    gradient_predictions,
    zero_division=0
)

gradient_confusion_matrix = confusion_matrix(
    y_test_gradient,
    gradient_predictions
)

print("Gradient Boosting Accuracy:", gradient_accuracy)
print("Gradient Boosting Precision:", gradient_precision)
print("Gradient Boosting Recall:", gradient_recall)
print("Gradient Boosting F1-score:", gradient_f1)
print("Gradient Boosting Confusion Matrix:")
print(gradient_confusion_matrix)

# Gradient Boosting confusion matrix visualization

plt.figure(figsize=(6, 5))
plt.imshow(gradient_confusion_matrix)

plt.title("Gradient Boosting Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            gradient_confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("static/gradient_confusion_matrix.png")
plt.close()

print("Gradient Boosting confusion matrix created successfully")

# Logistic Regression training

X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
    X_logistic,
    y_logistic,
    test_size=0.20,
    random_state=42,
    stratify=y_logistic
)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

logistic_predictions = logistic_model.predict(X_test_logistic)


print("Logistic Regression model trained successfully")
print("Training records:", len(X_train_logistic))
print("Testing records:", len(X_test_logistic))

# Logistic Regression evaluation

logistic_accuracy = accuracy_score(
    y_test_logistic,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test_logistic,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test_logistic,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test_logistic,
    logistic_predictions,
    zero_division=0
)

logistic_confusion_matrix = confusion_matrix(
    y_test_logistic,
    logistic_predictions
)

print("Accuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1-score:", logistic_f1)
print("Confusion Matrix:")
print(logistic_confusion_matrix)

# Confusion matrix visualization

plt.figure(figsize=(6, 5))
plt.imshow(logistic_confusion_matrix)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            logistic_confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("static/logistic_confusion_matrix.png")
plt.close()

print("Logistic Regression confusion matrix created successfully")

# Routes

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/machine-learning/concepts")
def ml_concepts():
    return render_template("ml_concepts.html")


@app.route("/machine-learning/types")
def ml_types():
    return render_template("ml_types.html")


@app.route("/use-cases/1")
def use_case_1():
    return render_template("use_case_1.html")


@app.route("/use-cases/2")
def use_case_2():
    return render_template("use_case_2.html")


@app.route("/use-cases/3")
def use_case_3():
    return render_template("use_case_3.html")


@app.route("/use-cases/4")
def use_case_4():
    return render_template("use_case_4.html")


@app.route("/supervised/linear-regression/concepts")
def regression_concepts():
    return render_template("regression_concepts.html")


@app.route("/supervised/logistic-regression/concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")

@app.route("/supervised/logistic-regression/evaluation")
def logistic_regression_evaluation():

    return render_template(
        "logistic_regression_evaluation.html",
        accuracy=logistic_accuracy,
        precision=logistic_precision,
        recall=logistic_recall,
        f1=logistic_f1
    )


@app.route(
    "/supervised/linear-regression/application",
    methods=["GET", "POST"]
)
def regression_application():

    prediction = None
    formatted_prediction = None
    error = None
    house_area = None

    if request.method == "POST":

        try:
            house_area = float(request.form["house_area"])

            if house_area <= 0:
                error = "Please enter a value greater than 0."

            else:
                prediction = linear_model.predict(
                    [[house_area]]
                )[0]

                formatted_prediction = "{:,.0f}".format(
                    prediction
                )

        except (ValueError, TypeError):
            error = "Please enter a valid numeric value."

    return render_template(
        "regression_application.html",
        prediction=prediction,
        formatted_prediction=formatted_prediction,
        error=error,
        house_area=house_area,
        r2=r2,
        coefficient=coefficient,
        intercept=intercept
    )

@app.route("/supervised/gradient-boosting/concepts")
def gradient_boosting_concepts():
    return render_template("gradient_boosting_concepts.html")

@app.route("/supervised/gradient-boosting/evaluation")
def gradient_boosting_evaluation():
    return render_template(
        "gradient_boosting_evaluation.html",
        accuracy=gradient_accuracy,
        precision=gradient_precision,
        recall=gradient_recall,
        f1=gradient_f1
    )

@app.route(
    "/supervised/gradient-boosting/application",
    methods=["GET", "POST"]
)
def gradient_boosting_application():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            house_area = float(request.form["house_area"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            house_age = int(request.form["house_age"])

            if house_area <= 0:
                error = "House area must be greater than 0."

            elif bedrooms <= 0:
                error = "Bedrooms must be greater than 0."

            elif bathrooms <= 0:
                error = "Bathrooms must be greater than 0."

            elif house_age < 0:
                error = "House age cannot be negative."

            else:
                input_data = [[
                    house_area,
                    bedrooms,
                    bathrooms,
                    house_age
                ]]

                prediction = gradient_model.predict(
                    input_data
                )[0]

                probability = gradient_model.predict_proba(
                    input_data
                )[0][prediction]

        except (ValueError, TypeError):
            error = "Please enter valid numeric values."

    return render_template(
        "gradient_boosting_application.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


@app.route(
    "/supervised/logistic-regression/application",
    methods=["GET", "POST"]
)
def logistic_regression_application():

    prediction = None
    probability = None
    house_area = None
    error = None

    if request.method == "POST":
        try:
            house_area = float(request.form["house_area"])

            if house_area <= 0:
                error = "Please enter a value greater than 0."

            else:
                prediction = logistic_model.predict(
                    [[house_area]]
                )[0]

                probability = logistic_model.predict_proba(
                    [[house_area]]
                )[0][prediction]

        except (ValueError, TypeError):
            error = "Please enter a valid numeric value."

    return render_template(
        "logistic_regression_application.html",
        prediction=prediction,
        probability=probability,
        house_area=house_area,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)