from __future__ import print_function

import math
import random
from collections import namedtuple, Counter

__AUTHOR__ = 'Vyas Alwar'

# FUNCTIONS #


def most_common_value(lst):
    """Returns the most common value in a binary iterable"""
    return sum(lst) / len(lst) > .5


def entropy(lst):
    """Calculates the entropy of a binary iterable"""
    if all(lst) or not any(lst):
        return 0

    x_true = lst.count(True) / len(lst)
    x_false = lst.count(False) / len(lst)

    return -x_true * math.log(x_true, 2) - x_false * math.log(x_false, 2)


def information_gain(vector_collection, attribute):
    """Calculates the information gain in a collection of vectors when separated
    by an attribute"""
    initial_entropy = entropy([vector.CLASS for vector in vector_collection])

    pos_attr = [v for v in vector_collection if v.attributes[attribute]]
    pos_entropy = entropy([vector.CLASS for vector in pos_attr])
    pos_value = pos_entropy * len(pos_attr) / len(vector_collection)

    neg_attr = [v for v in vector_collection if not v.attributes[attribute]]
    neg_entropy = entropy([vector.CLASS for vector in neg_attr])
    neg_value = neg_entropy * len(neg_attr) / len(vector_collection)

    return initial_entropy - pos_value - neg_value


def best_classifier(vector_collection, attributes):
    """Determines which attribute is the best classifier of the
    vector_collection, determined by information gain"""
    def compare_attributes(attribute):
        return information_gain(vector_collection, attribute)

    best_attribute = max(attributes, key=compare_attributes)
    return best_attribute


def partition(dataset, training_size):
    """Partitions the dataset into training and testing data"""
    random.shuffle(dataset)
    return dataset[:training_size], dataset[training_size:]

# Classes


Vector = namedtuple('Vector', ['attributes', 'CLASS'])


class Node():
    """A simple class to represent a node in a tree or digraph."""
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.links = [None, None]
        self.terminal = True
        self.terminal_classification = None

    def __repr__(self):
        return "Root: " + self.pprint()

    def pprint(self, level=0, branch=None):
        "Pretty printing for trees"
        ret = "\t"*level

        if branch == 1:
            ret += "False branch: "
        elif branch == 0:
            ret += "True branch: "
        if self.attribute:
            ret += str(self.attribute)+"\n"
        else:
            ret += 'CLASSIFY AS: ' + str(self.terminal_classification) + "\n"

        for i, child in enumerate(reversed(self.links)):
            if child:
                ret += child.pprint(level+1, i)
        return ret

    def link(self, other_node, value):
        "Stores references to other nodes in this node"
        if value is True:
            self.links[True] = other_node
            self.terminal = False
        if value is False:
            self.links[False] = other_node
            self.terminal = False

    def classify_terminal_node(self, value):
        "If the node is a terminal node, assign it a terminal classification"
        if self.terminal:
            self.terminal_classification = value


class Classifier():
    "Abstract class that provides an interface for some classifier"

    def __init__(self, labels, training_set, testing_set):
        self.labels = labels
        self.training_set = training_set
        self.testing_set = testing_set

    def train(self):
        "Train the classifer on the training and testing sets"
        raise NotImplementedError()

    def classify(self, attributes):
        "Classify a list of attributes"
        raise NotImplementedError()

    def test(self):
        """Evaluates the decision tree on the training set."""
        count = Counter()
        for example in self.testing_set:
            classification = self.classify(example.attributes)

            if example.CLASS and classification:
                count['TP'] += 1
            elif not example.CLASS and classification:
                count['FP'] += 1
            elif not example.CLASS and not classification:
                count['TN'] += 1
            elif example.CLASS and not classification:
                count['FN'] += 1
        return count


class PriorClassifier(Classifier):
    "A baseline classifier that simply returns the mode of the training set"

    def __init__(self, labels, training_set, testing_set):
        super().__init__(labels, training_set, testing_set)
        self.classification = self.train()

    def train(self):
        return most_common_value([vector.CLASS for vector in self.training_set])

    def classify(self, attributes):
        return self.classification


class DecisionTree(Classifier):
    "A decision tree classifier"

    def __init__(self, labels, training_set, testing_set):
        super(DecisionTree, self).__init__(labels, training_set, testing_set)
        self.tree = self.train(self.training_set, self.labels)

    def __repr__(self):
        return str(t.tree)

    def train(self, examples, labels):
        """Implements the id3 algorithm to determine the structure of a decision
        tree. Returns the tree itself."""
        root = Node()

        if all([datum.CLASS for datum in examples]):
            root.classify_terminal_node(True)

        elif all([not datum.CLASS for datum in examples]):
            root.classify_terminal_node(False)

        elif not labels:
            average = most_common_value([datum.CLASS for datum in examples])
            root.classify_terminal_node(average)

        else:
            splitter = best_classifier(examples, labels)
            new_labels = [label for label in labels if label != splitter]

            pos_branch = [v for v in examples if v.attributes[splitter]]
            neg_branch = [v for v in examples if not v.attributes[splitter]]

            root.attribute = splitter
            root.link(self.train(pos_branch, new_labels), True)
            root.link(self.train(neg_branch, new_labels), False)

        return root

    def classify(self, attributes):
        """Classifies the given case according to the decision tree"""
        cur_tree = self.tree
        while not cur_tree.terminal:
            branch = attributes[cur_tree.attribute]
            cur_tree = cur_tree.links[branch]
        return cur_tree.terminal_classification
