import argparse
import csv

from decisiontree import Vector, DecisionTree, PriorClassifier, partition


def parse_file(filename):
    """Reads and processes the input file"""
    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t')
        labels = next(reader)
        labels = labels[:-1]
        parsed_data = []
        for row in reader:
            boolean_row = [val == 'true' for val in row]
            CLASS = boolean_row[-1]
            attributes = {labels[i]: boolean_row[i]
                    for i in range(len(boolean_row) - 1)}
            parsed_data.append(Vector(attributes, CLASS))
    return labels, parsed_data

def get_parser():
    parser = argparse.ArgumentParser('Runs a decision tree classifier on a dataset')
    parser.add_argument('input_file_name', type=str)
    parser.add_argument('number_of_trials', type=int)
    parser.add_argument('training_set_size', type=int)
    parser.add_argument('--verbose', type=bool)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_file_name = args.input_file_name
    number_of_trials = args.number_of_trials
    training_set_size = args.training_set_size
    verbose = args.verbose


    decision_tree_performance = []
    prior_classifier_performance = []

    for i in range(number_of_trials):
        labels,dataset = parse_file(input_file_name)
        training_set, testing_set = partition(dataset, training_set_size)
        testing_set_size = len(testing_set)

        t = DecisionTree(labels, training_set, testing_set)
        p = PriorClassifier(labels, training_set, testing_set)

        # Evaluating Testing set performance
        c = t.test()
        tree_prob = int(round(100 * float(c['TP'] + c['TN']) /
            len(t.testing_set)))
        decision_tree_performance.append(tree_prob)

        k = p.test()
        prior_prob = int(round(100. * float(k['TP'] + k['TN']) /
            len(p.testing_set)))
        prior_classifier_performance.append(prior_prob)


# PRINTING ALL THE THINGS ####

        print("""
TRIAL NUMBER: %d
--------------------

DECISION TREE STRUCTURE: """ % i)
        print(t.tree)
        print("""

Percent of test cases correctly classified by a decision tree built 
\twith ID3 = %d%%

Percent of test cases correctly classified by using prior 
\tprobabilities from the training set = %d%% """ % (tree_prob, prior_prob))

        if verbose:
            print("Examples in the training set\n--------------------""")
            print("\t".join(label for label in t.labels) + '\n')

            for vector in t.training_set:
                example_string = str(vector.attributes[t.labels[0]])
                for i in range(len(t.labels) - 1):
                    tabs = '\t' * ((len(t.labels[i]) // 8) + 1)
                    label = t.labels[i+1]
                    example_string += tabs + str(vector.attributes[label])
                print(example_string)

            print("Examples in the testing set\n--------------------")
            labels = t.labels + ["CLASS", "PRIOR RESULT", "ID3 RESULT"]

            print("\t".join(label for label in labels) + '\n')
            for vector in t.testing_set:
                example_string = str(vector.attributes[t.labels[0]])
                for j in range(len(t.labels) - 1):
                    tabs = '\t' * ((len(t.labels[j]) // 8) + 1)
                    label = t.labels[j+1]
                    example_string += tabs + str(vector.attributes[label])

                last_tab = '\t' * ((len(t.labels[-1]) // 8) + 1)
                example_string += last_tab + str(vector.CLASS) + '\t'
                example_string += str(False) + '\t\t'
                example_string += str(t.classify(vector.attributes))
                print(example_string)

        treeMean = sum(decision_tree_performance) / len(decision_tree_performance)
        priorMean = sum(prior_classifier_performance) / len(prior_classifier_performance)

    print("""
Example file used = %s
Number of trials = %d
Training set size for each trial = %d
Testing set size for each trial = %d
Mean performance of decision tree over all trials = %d%%
Mean performance of using prior probability derived from training set = %d%%
    correct classification
""" % (input_file_name, number_of_trials, training_set_size, testing_set_size,
        treeMean, priorMean))


if __name__ == '__main__':
    main()
