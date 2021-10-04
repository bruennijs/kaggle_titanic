import string

from graphviz import Source
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def plot_classifier(clf: DecisionTreeClassifier, *, feature_names: list[string] = None, class_names: list[string] = None):
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf,
                       feature_names=feature_names,
                       class_names=class_names,
                       filled=True)
    return



def plot_clf_file(clf, feature_names=None, class_names=None) -> Source:

    export_graphviz(clf, out_file="tree.dot", class_names=class_names,
                feature_names=feature_names, impurity=False, filled=True)

    with open("tree.dot") as f:
        dot_graph = f.read()
        return Source(dot_graph)

