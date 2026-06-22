import pandas as pd

# Find-S Algorithm
def find_s_algorithm(file_path):

    # Read CSV file
    data = pd.read_csv(file_path)

    print("Training Data:")
    print(data)

    # Separate attributes and target column
    attributes = data.columns[:-1]
    target = data.columns[-1]

    # Initialize hypothesis with None
    hypothesis = None

    # Process only positive examples
    for index, row in data.iterrows():

        if row[target] == "Yes":

            if hypothesis is None:
                hypothesis = list(row[attributes])

            else:
                for i in range(len(hypothesis)):

                    if hypothesis[i] != row[attributes][i]:
                        hypothesis[i] = "?"

    return hypothesis


# Main Program
file_path = "training_data.csv"

final_hypothesis = find_s_algorithm(file_path)

print("\nFinal Hypothesis:")
print(final_hypothesis)
