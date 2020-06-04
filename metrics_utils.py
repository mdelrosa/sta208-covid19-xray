from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

import matplotlib.pyplot as plt

def calc_roc(y_test,y_hat):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_hat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return [fpr,tpr,roc_auc,n_classes]
    
def plot_roc(fpr,tpr,roc_auc,n_classes,C,vect_name="tfidf"):
    plt.figure()
    class_names = ["earn","acq"]
    class_colors = ["darkgoldenrod","darkslategrey","darkslateblue"]
    lw = 2
    for i in range(n_classes-1):
        plt.plot(fpr[i], tpr[i], color=class_colors[i],
             lw=lw, label='{} (area = {:0.2f})'.format(class_names[i],roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='grey', lw=lw/2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('roc for {} (C={})'.format(vect_name,C))
    plt.legend(loc="lower right")
    plt.show()
    
def calc_pr(y_test,y_hat):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_hat[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_hat[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_hat.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_hat,
                                                         average="micro")
    return [precision,recall,average_precision,n_classes]
    
def plot_pr(precision,recall,average_precision,n_classes,C,classifier_name="ResNet50"):
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    class_names = ["earn","acq"]
    for i, color in zip(range(n_classes-1), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for {} (area = {:0.2f})'
                      ''.format(class_names[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve for {} (C={})'.format(classifier_name,C))
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()