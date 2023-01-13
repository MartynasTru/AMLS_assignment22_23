
def logging(runtime, model, report, best_parameter, best_model):
    with open("A1/logs.txt", "a") as external_file:
        print("Runtime: ", runtime, file=external_file)
        print("Used Classifier: ", model, file=external_file)
        print("Best parameters: ", best_parameter, file=external_file)
        print("best models performance (accuracy): ", best_model, file=external_file)
        print("Report: ", report, file=external_file)
        print("\n")

        external_file.close()


def model_classification(report):
    with open("A1/logs.txt", "a") as external_file:
        print("\n")
        print(report, file=external_file)
        print("\n")

        external_file.close()
