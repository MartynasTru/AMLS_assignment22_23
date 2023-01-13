import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_search(cv_results, model, conf_matrix, train_sizes, train_scores, val_scores):

    model_name = model
    if model_name == "SVM":

        #Parameter C performance
        scores_mean = cv_results["mean_test_score"] * 100
        param_C = [result["C"] for result in cv_results["params"]]
        plt.rcParams["figure.figsize"] = [15, 10]
        plt.scatter(param_C, scores_mean, label="Test Score vs C", s=80, alpha=1)
        plt.grid(which="both", linestyle=":", linewidth=1.5)
        plt.xlabel("Parameter C tunning", fontsize=27)
        plt.ylabel("Accuracy Score, %", fontsize=27)
        plt.tick_params(axis="both", which="major", labelsize=16.5)
        plt.title("SVM model. Linear kernel. C performance", fontsize=32)
        plt.savefig("A1/graphs/SVC model C performance, linear.png")

        # Parameter Gamma performance
        scores_mean = cv_results["mean_test_score"] * 100
        param_gamma = [result["gamma"] for result in cv_results["params"]]
        plt.rcParams["figure.figsize"] = [15, 10]
        plt.scatter(param_gamma, scores_mean, label="Test Score vs C", s=80, alpha=1)
        plt.grid(which="both", linestyle=":", linewidth=1.5)
        plt.xlabel("Parameter C tunning", fontsize=27)
        plt.ylabel("Accuracy Score, %", fontsize=27)
        plt.tick_params(axis="both", which="major", labelsize=16.5)
        plt.title("SVM model. Linear kernel. Gamma performance", fontsize=32)
        plt.savefig("A1/graphs/SVC model gamma performance.png")

        # Confusion matrix heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            cmap="Blues",
            fmt="d",
            annot_kws={"size": 20, "color": "red"},
            cbar_kws={"ticks": [0, 1], "label": "My Colorbar"},
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SVM model confusion matrix", fontsize=32)
        plt.savefig("A1/graphs/SVM confusion matrix.png")

        # Learning curve plot
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")
        plt.xlabel("Number of Training Examples")
        plt.ylabel("Score")
        plt.legend()
        plt.title("SVM learning curve", fontsize=32)
        plt.savefig("A1/graphs/KNN learningcurve.png")

    elif model_name == "KNN":

        # Parameter n_neighbors performance
        scores_mean = cv_results["mean_test_score"] * 100
        param_n = [result["n_neighbors"] for result in cv_results["params"]]
        plt.rcParams["figure.figsize"] = [15, 10]
        plt.scatter(param_n, scores_mean, label="Test Score vs n", s=80, alpha=1)
        plt.grid(which="both", linestyle=":", linewidth=1.5)
        plt.xlabel("Number of neighbours, N", fontsize=27)
        plt.ylabel("Accuracy Score, %", fontsize=27)
        plt.tick_params(axis="both", which="major", labelsize=16.5)
        plt.title("KNN Regressiom model. N performance", fontsize=32)
        plt.savefig("A1/graphs/KNN model N performance, linear.png")

    elif model_name == "RF":

        # Parameter n_estimators performance
        scores_mean = cv_results["mean_test_score"] * 100
        param_n = [result["n_estimators"] for result in cv_results["params"]]
        plt.rcParams["figure.figsize"] = [15, 10]
        plt.scatter(param_n, scores_mean, label="Test Score vs n", s=80, alpha=1)
        plt.grid(which="both", linestyle=":", linewidth=1.5)
        plt.xlabel("Number of estimators, N", fontsize=27)
        plt.ylabel("Accuracy Score, %", fontsize=27)
        plt.tick_params(axis="both", which="major", labelsize=16.5)
        plt.title("Random Forest model. N_estimators performance", fontsize=32)
        plt.savefig("A1/graphs/RF model N_estimators performance.png")