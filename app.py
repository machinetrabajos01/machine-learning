from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score,
    f1_score,
    silhouette_score
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

# Manual K-Means Exercise

manual_student_data = pd.read_csv(
    "data/student_performance_manual_100.csv"
)

manual_features = manual_student_data[
    [
        "Study_Hours",
        "Exam_Score"
    ]
].to_numpy()

# Define the three initial centroids

manual_centroids = np.array([
    [3.0, 45.0],
    [7.0, 70.0],
    [11.0, 90.0]
])

manual_iterations = []

# Perform three manual K-Means iterations

for iteration in range(1, 4):

    # Calculate Euclidean distances

    distances = np.sqrt(
        (
            manual_features[:, np.newaxis, :]
            - manual_centroids[np.newaxis, :, :]
        ) ** 2
    ).sum(axis=2)

    # Assign each student to the nearest centroid

    assignments = np.argmin(
        distances,
        axis=1
    )

    # Calculate updated centroids

    updated_centroids = []

    for cluster in range(3):

        cluster_points = manual_features[
            assignments == cluster
        ]

        if len(cluster_points) > 0:

            new_centroid = cluster_points.mean(
                axis=0
            )

        else:

            new_centroid = manual_centroids[
                cluster
            ]

        updated_centroids.append(
            new_centroid
        )

    updated_centroids = np.array(
        updated_centroids
    )

    # Calculate within-cluster variance

    cluster_variances = []

    for cluster in range(3):

        cluster_distances = distances[
            assignments == cluster,
            cluster
        ]

        if len(cluster_distances) > 0:

            variance = np.mean(
                cluster_distances ** 2
            )

        else:

            variance = 0.0

        cluster_variances.append(
            variance
        )

    total_variance = sum(
        cluster_variances
    )

    # Store iteration results

    iteration_result = {
        "iteration": iteration,
        "distances": distances.copy(),
        "assignments": assignments.copy(),
        "centroids_before": manual_centroids.copy(),
        "centroids_after": updated_centroids.copy(),
        "cluster_variances": cluster_variances,
        "total_variance": total_variance
    }

    manual_iterations.append(
        iteration_result
    )

    # Update centroids for the next iteration

    manual_centroids = updated_centroids


print("Manual K-Means exercise loaded successfully")
print("Manual dataset records:", len(manual_student_data))

for result in manual_iterations:

    print(
        "Iteration:",
        result["iteration"]
    )

    print(
        "Updated centroids:"
    )

    print(
        result["centroids_after"]
    )

    print(
        "Total within-cluster variance:",
        result["total_variance"]
    )

# Activity 3 - Unsupervised Machine Learning

student_data = pd.read_csv(
    "data/student_performance_1200.csv"
)

print("Student performance dataset loaded successfully")
print("Student records:", len(student_data))
print(student_data.head())

# Select the numerical variables for clustering

X_student = student_data[
    [
        "Average_Assessment_Score",
        "Total_VLE_Clicks"
    ]
]

print("Clustering variables selected successfully")
print("Clustering variables:", list(X_student.columns))

# Standardize the clustering variables

student_scaler = StandardScaler()

X_student_scaled = student_scaler.fit_transform(
    X_student
)

print("Clustering variables standardized successfully")

# Configure and train the K-Means model

student_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

student_clusters = student_kmeans.fit_predict(
    X_student_scaled
)

print("K-Means model trained successfully")
print("Number of clusters:", student_kmeans.n_clusters)

# Calculate clustering evaluation metrics

student_silhouette = silhouette_score(
    X_student_scaled,
    student_clusters
)

student_centroids_scaled = student_kmeans.cluster_centers_

print("Silhouette Score:", student_silhouette)
print("Scaled cluster centroids:")
print(student_centroids_scaled)

# Convert centroids back to the original scale

student_centroids = student_scaler.inverse_transform(
    student_centroids_scaled
)

print("Cluster centroids in original scale:")
print(student_centroids)

# Assign cluster labels to the student dataset

student_data["Cluster"] = student_clusters

print("Cluster labels assigned successfully")
print(student_data.head())

# Create a dataframe with cluster centroid information

student_centroid_table = pd.DataFrame(
    student_centroids,
    columns=[
        "Average_Assessment_Score",
        "Total_VLE_Clicks"
    ]
)

student_centroid_table["Cluster"] = range(
    len(student_centroid_table)
)

print("Cluster centroid table created successfully")
print(student_centroid_table)

# Count students in each cluster

student_cluster_counts = (
    student_data["Cluster"]
    .value_counts()
    .sort_index()
)

print("Students per cluster:")
print(student_cluster_counts)

# Create the final cluster summary table

student_cluster_summary = student_centroid_table.copy()

student_cluster_summary["Student_Count"] = (
    student_cluster_counts.values
)

student_cluster_summary["Cluster_Percentage"] = (
    student_cluster_summary["Student_Count"]
    / len(student_data)
    * 100
)

print("Cluster summary table created successfully")
print(student_cluster_summary)

# Create the K-Means cluster visualization

plt.figure(figsize=(10, 6))

for cluster in sorted(student_data["Cluster"].unique()):
    cluster_data = student_data[
        student_data["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Average_Assessment_Score"],
        cluster_data["Total_VLE_Clicks"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.scatter(
    student_centroids[:, 0],
    student_centroids[:, 1],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Average Assessment Score")
plt.ylabel("Total VLE Clicks")
plt.title("Student Performance Clusters")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "static/student_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Student clustering plot created successfully")

# Routes

# Unsupervised Machine Learning routes

@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised_concepts.html")


@app.route("/unsupervised/manual-exercise")
def unsupervised_manual_exercise():

    manual_records = manual_student_data.to_dict(
        orient="records"
    )

    manual_iteration_results = []

    for result in manual_iterations:

        distances = result["distances"]

        iteration_records = []

        for index, record in manual_student_data.iterrows():

            iteration_records.append({
                "Student_ID": int(record["Student_ID"]),
                "Study_Hours": float(record["Study_Hours"]),
                "Exam_Score": float(record["Exam_Score"]),
                "Distance_Cluster_0": float(
                    distances[index][0]
                ),
                "Distance_Cluster_1": float(
                    distances[index][1]
                ),
                "Distance_Cluster_2": float(
                    distances[index][2]
                ),
                "Assigned_Cluster": int(
                    result["assignments"][index]
                )
            })

        manual_iteration_results.append({
            "iteration": result["iteration"],
            "records": iteration_records,
            "centroids_before": result["centroids_before"].tolist(),
            "centroids_after": result["centroids_after"].tolist(),
            "cluster_variances": result["cluster_variances"],
            "total_variance": result["total_variance"]
        })

    return render_template(
        "unsupervised_manual_exercise.html",
        manual_records=manual_records,
        manual_iterations=manual_iteration_results
    )

# Create plots for the manual K-Means iterations

manual_variances = []

for result in manual_iterations:

    iteration = result["iteration"]
    assignments = result["assignments"]
    centroids = result["centroids_after"]

    plt.figure(figsize=(10, 6))

    for cluster in range(3):

        cluster_points = manual_features[
            assignments == cluster
        ]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster}",
            alpha=0.6
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=250,
        label="Centroids"
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")

    plt.title(
        f"Manual K-Means - Iteration {iteration}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"static/student_kmeans_iteration_{iteration}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    manual_variances.append(
        result["total_variance"]
    )


# Create variance comparison plot

plt.figure(figsize=(10, 6))

iterations = [
    result["iteration"]
    for result in manual_iterations
]

plt.plot(
    iterations,
    manual_variances,
    marker="o"
)

plt.xlabel("Iteration")
plt.ylabel("Total Within-Cluster Variance")

plt.title(
    "Within-Cluster Variance Across K-Means Iterations"
)

plt.xticks(iterations)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "static/student_kmeans_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Manual K-Means plots created successfully"
)


# Activity 3 - Unsupervised Machine Learning routes

@app.route("/unsupervised/clustering-application")
def unsupervised_clustering_application():

    student_records = student_data.to_dict(
        orient="records"
    )

    cluster_summary = student_cluster_summary.to_dict(
        orient="records"
    )

    return render_template(
        "clustering_application.html",
        student_records=student_records,
        cluster_summary=cluster_summary,
        silhouette_score=student_silhouette
    )

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