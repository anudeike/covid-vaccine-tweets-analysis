import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import itertools

# HELPER FUNCTIONS
# id,CAP,astroturf,fake_follower,financial,other,overall,self-declared,spammer,type
def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))], average="weighted")
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))], average='weighted')
    baseline['roc'] = 0.5

    results = {}

    results['recall'] = recall_score(test_labels, predictions, average="weighted")
    results['precision'] = precision_score(test_labels, predictions, average="weighted")
    results['roc'] = roc_auc_score(test_labels, probs, multi_class="ovr", average="weighted")

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions, average="weighted")
    train_results['precision'] = precision_score(train_labels, train_predictions, average="weighted")
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()

# confusion matrix function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    plt.show()
# path to master class
master_training_set_path = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot.csv"

# read the csv
df = pd.read_csv(master_training_set_path)

# get the values as an array
labels = np.array(df["labels"].values)

# get rid of labels in the training set
df.drop(['labels'],axis=1,inplace=True)

# get rid of id -> has nothing to do with output in the training set
df.drop(['id'],axis=1,inplace=True)

# split data for training
train, test, train_labels, test_labels = train_test_split(df, labels,
                                                          stratify=labels,
                                                          test_size=0.3)

# create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')


# train
model.fit(train, train_labels)

# get the amount of nodes are in tree on average and the max depth. about 100 trees total
n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

#evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

# Calculate roc auc
#roc_value = roc_auc_score(test_labels, rf_probs)
#print(f'ROC VALUE: {roc_value}')


# Extract feature importances
fi = pd.DataFrame({'feature': list(train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

# Display
print(fi.head())

# show the confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
# plot_confusion_matrix(cm, classes = ['Not a Bot', 'Is A Bot'],
#                       title = 'Bot Confusion Matrix')
#print(f'Confusion Matrix: {cm}')

