import string

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def plot_classifier(clf: DecisionTreeClassifier, *, feature_names: list[string] = None, class_names: list[string] = None):
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf,
                       feature_names=feature_names,
                       class_names=class_names,
                       filled=True)
    return
