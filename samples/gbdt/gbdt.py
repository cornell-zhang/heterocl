from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import heterocl as hcl
import numpy as np

iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

params = {'n_estimators': 6, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01}
gbc = GradientBoostingClassifier(**params)
gbc.fit(X_train, y_train)

print("Accuracy (training): {0:.3f}".format(gbc.score(X_train, y_train)))
print("Accuracy (validation): {0:.3f}".format(gbc.score(X_test, y_test)))

# generte raw prediction
fnames = iris_dataset.feature_names
num_feat = len(fnames)

# create heterocl program 
num_sample = len(X_test)
num_tree, num_class = gbc.estimators_.shape

dtype = hcl.Float()
hcl.init(hcl.Float())
inputs = hcl.placeholder((num_sample, num_feat), "inputs")

def kernel(inputs):
    pred_mat = hcl.compute((num_sample, num_class), lambda *args: 0, "pred")

    def tree_to_code(tree, feature_names, k, x):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        print("\n  # tree ({}):".format(k, ", ".join(feature_names)))

        def recurse(node, depth):
            indent = depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                with hcl.if_(inputs[x, feature_name.index(name)] <= threshold):
                    print("{}with hcl.if_(inputs[{}] <= {}):".format(indent * "    ", \
                            x*num_class + feature_name.index(name), threshold))
                    recurse(tree_.children_left[node], depth + 1)
                with hcl.else_():
                    print("{}with hcl.else_():".format(indent * "    "))
                    recurse(tree_.children_right[node], depth + 1)
            else:
                # print(tree_.value[node], float(tree_.value[node]))
                pred_mat[x, k] += float(tree_.value[node])
                print("{}pred_mat[{}] += {}".format(indent * "    ", \
                        x*num_class + k, float(tree_.value[node])))
        # root node
        recurse(0, 1)

    def update(x):
        # iterate through examples 
        for i in range(num_tree):
          for k in range(num_class):
            # update the predication matrix for each class
            tree_to_code(gbc.estimators_[i, k], fnames, k, x)

    hcl.mutate((num_sample,), lambda x: update(x), "update")
    return pred_mat

target = "llvm"
s = hcl.create_schedule([inputs], kernel)
# print(hcl.lower(s))

f = hcl.build(s, target)
input_hcl = hcl.asarray(X_test, dtype=dtype)
output_hcl = hcl.asarray(np.zeros((num_sample, num_class)), dtype=dtype)
f(input_hcl, output_hcl)

print(output_hcl.asnumpy())
