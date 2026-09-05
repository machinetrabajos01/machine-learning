from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
data = pd.read_csv("data/housing_dataset.csv")

print("Dataset loaded successfully")
print("Number of records:", len(data))
print(data.head())

X = data[["House_Area"]]
y = data["House_Price"]

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

r2 = r2_score(y, predictions)

coefficient = model.coef_[0]

intercept = model.intercept_

print("Linear Regression model trained successfully")
print("R² Score:", r2)
print("Coefficient:", coefficient)
print("Intercept:", intercept)

model.fit(X, y)

print("Linear Regression model trained successfully")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.figure(figsize=(10, 6))

plt.scatter(
    data["House_Area"],
    data["House_Price"],
    alpha=0.6
)

plt.plot(
    data["House_Area"],
    model.predict(X)
)

plt.title("House Area vs House Price")
plt.xlabel("House Area (m²)")
plt.ylabel("House Price (COP)")

plt.grid(True)

plt.tight_layout()

plt.savefig("static/housing_regression.png")

plt.close()

print("Regression plot created successfully")


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

                prediction = model.predict(
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


if __name__ == "__main__":
    app.run(debug=True)