from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score, geometric_mean_score, make_index_balanced_accuracy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.multiclass import unique_labels


def resultCsv(filepath, X, trueY, predY):
    df = pd.DataFrame(X)
    df['TrueY'] = trueY
    df['PredY'] = predY
    df.to_csv(filepath, index=False)


def classificationReportDict(trueY, predY,
                             labels=None,
                             targetNames=None,
                             sampleWeight=None,
                             alpha=0.1):
    report = dict()
    if labels is None:
        labels = unique_labels(trueY, predY)
    else:
        labels = np.asarray(labels)
    if targetNames is None:
        targetNames = [str(label) for label in labels]
    # Precision Recall F1 Support
    precision, recall, f1, support = \
        precision_recall_fscore_support(trueY, predY, labels=labels, average=None, sample_weight=sampleWeight)
    # Specificity
    specificity = specificity_score(trueY, predY, labels=labels, average=None, sample_weight=sampleWeight)
    # Geometric mean
    gMean = geometric_mean_score(trueY, predY, labels=labels, average=None, sample_weight=sampleWeight)
    # Index balanced accuracy
    ibaGMeanScore = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)
    ibaGMean = ibaGMeanScore(trueY, predY, labels=labels, average=None, sample_weight=sampleWeight)
    for i, label in enumerate(labels):
        targetName = targetNames[i]
        report[targetName] = {
            'Precision': precision[i],
            'Recall': recall[i],
            'F1': f1[i],
            'Specificity': specificity[i],
            'GMean': gMean[i],
            'IbaGMean': ibaGMean[i],
            'Support': support[i],
        }

    report['Weighted Avg'] = {
        'Precision': np.average(precision, weights=support),
        'Recall': np.average(recall, weights=support),
        'F1': np.average(f1, weights=support),
        'Specificity': np.average(specificity, weights=support),
        'GMean': np.average(gMean, weights=support),
        'IbaGMean': np.average(ibaGMean, weights=support),
        'Support': np.sum(support)
    }

    report['Macro Avg'] = {
        'Precision': np.average(precision),
        'Recall': np.average(recall),
        'F1': np.average(f1),
        'Specificity': np.average(specificity),
        'GMean': np.average(gMean),
        'IbaGMean': np.average(ibaGMean),
        'Support': np.sum(support)
    }

    # Accuracy
    accuracy = accuracy_score(trueY, predY, normalize=True, sample_weight=sampleWeight)
    report['Accuracy'] = accuracy

    return report


def resultReport(filepath, trueY, predY, labels=None):
    labels = unique_labels(trueY, predY) if labels is None else np.asarray(labels)
    report = classificationReportDict(trueY, predY, labels=labels)
    report[''] = {}
    report['Accuracy'] = {'Precision': report.pop('Accuracy')}
    report = pd.DataFrame(report).transpose()
    report.to_excel(filepath)


def resultConfusionMatrix(filepath, trueY, predY, labels=None):
    labels = unique_labels(trueY, predY) if labels is None else np.asarray(labels)
    confusionMatrix = confusion_matrix(trueY, predY, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(confusionMatrix, display_labels=labels). \
        plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(filepath)


def y2scores(y, labels=None):
    labels = unique_labels(y) if labels is None else np.asarray(labels)
    scores = np.zeros((y.shape[0], labels.shape[0]))
    for n, label in enumerate(labels):
        scores[y == label, n] = 1
    return scores, labels


def resultRocCurve(filepath, trueScoreY, predScoreY, labels=None):
    labels = np.arange(trueScoreY.shape[1]) if labels is None else np.asarray(labels)
    fpr, tpr, rocAuc = dict(), dict(), dict()
    for n, label in enumerate(labels):
        fpr[label], tpr[label], _ = roc_curve(trueScoreY[:, n], predScoreY[:, n])
        rocAuc[label] = auc(fpr[label], tpr[label])
    fpr['Micro Avg'], tpr['Micro Avg'], _ = roc_curve(trueScoreY.ravel(), predScoreY.ravel())
    rocAuc['Micro Avg'] = auc(fpr['Micro Avg'], tpr['Micro Avg'])
    allFpr = np.unique(np.concatenate([fpr[label] for label in labels]))
    meanTpr = np.zeros_like(allFpr)
    for label in labels:
        meanTpr += np.interp(allFpr, fpr[label], tpr[label])
    meanTpr /= labels.shape[0]
    fpr['Macro Avg'] = allFpr
    tpr['Macro Avg'] = meanTpr
    rocAuc['Macro Avg'] = auc(fpr['Macro Avg'], tpr['Macro Avg'])
    plt.figure(figsize=(16, 9))
    lw = 2
    fontSize = 20
    for key in fpr.keys():
        plt.plot(fpr[key], tpr[key], label=f'ROC {key} ({rocAuc[key]: .2f})', lw=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xlabel('False Positive Rate', fontsize=fontSize)
    plt.ylabel('True Positive Rate', fontsize=fontSize)
    plt.title('ROC Curve', fontsize=fontSize)
    plt.legend(loc='lower right', fontsize=fontSize)
    plt.savefig(filepath)


def resultPrecisionRecallCurve(filepath, trueScoreY, predScoreY, labels=None):
    labels = np.arange(trueScoreY.shape[1]) if labels is None else np.asarray(labels)
    precision, recall, ap = dict(), dict(), dict()
    for n, label in enumerate(labels):
        precision[label], recall[label], _ = precision_recall_curve(trueScoreY[:, n], predScoreY[:, n])
        ap[label] = average_precision_score(trueScoreY[:, n], predScoreY[:, n])
    plt.figure(figsize=(10, 10))
    lw = 2
    fontSize = 20
    for key in precision.keys():
        plt.plot(recall[key], precision[key], label=f'PR {key} (AP={ap[key]: .2f})', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xlabel('Recall', fontsize=fontSize)
    plt.ylabel('Precision', fontsize=fontSize)
    plt.title('Precision Recall Curve', fontsize=fontSize)
    plt.legend(loc='lower left', fontsize=fontSize)
    plt.savefig(filepath)


def results(name, trueY, predY, predScoreY=None, labels=None, X=None, folder='Data/Results'):
    labels = unique_labels(trueY, predY) if labels is None else np.asarray(labels)
    folder = Path(folder) / name
    folder.mkdir(parents=True, exist_ok=True)
    resultCsv(folder / f'{name}_Result.csv', X, trueY, predY)
    resultReport(folder / f'{name}_Report.xlsx', trueY, predY, labels)
    resultConfusionMatrix(folder / f'{name}_ConfusionMatrix.png', trueY, predY, labels)
    trueScoreY, _ = y2scores(trueY, labels)
    predScoreY, _ = y2scores(predY, labels) if predScoreY is None else (predScoreY, None)
    resultRocCurve(folder / f'{name}_RocCurve.png', trueScoreY, predScoreY, labels)
    resultPrecisionRecallCurve(folder / f'{name}_PrecisionRecallCurve.png', trueScoreY, predScoreY, labels)
