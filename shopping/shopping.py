import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    
    with open(filename, 'r') as file:
        data = csv.reader(file)
        next(data)
        
        for line in data:
            # Convert the values to the desired types
            administrative = int(line[0])
            administrative_duration = float(line[1])
            informational = int(line[2])
            informational_duration = float(line[3])
            product_related = int(line[4])
            product_related_duration = float(line[5])
            bounce_rates = float(line[6])
            exit_rates = float(line[7])
            page_values = float(line[8])
            special_day = float(line[9])
            month = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(line[10])
            operating_systems = int(line[11])
            browser = int(line[12])
            region = int(line[13])
            traffic_type = int(line[14])
            visitor_type = 1 if line[15] == "Returning_Visitor" else 0
            weekend = 1 if line[16] == "TRUE" else 0
            label = 1 if line[17] == "TRUE" else 0

            # Create an evidence list for the row
            line_evidence = [
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
                page_values,
                special_day,
                month,
                operating_systems,
                browser,
                region,
                traffic_type,
                visitor_type,
                weekend
            ]

            evidence.append(line_evidence)
            labels.append(label)

    return evidence, labels
        


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    k_nearest = model.fit(evidence, labels)
    return k_nearest


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_positive += 1
            if prediction == 1:
                true_positive += 1
        else:
            total_negative += 1
            if prediction == 0:
                true_negative += 1

    sensitivity = true_positive / total_positive
    specificity = true_negative / total_negative

    return sensitivity, specificity

if __name__ == "__main__":
    main()
