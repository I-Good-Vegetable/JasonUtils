from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle

from JasonUtils.Utils import tupleAppend


def loadNextCon2020(folder, shuffleDataset=False, randomState=None,
                    withLocalNormalTraffic=False):
    folder = Path(folder)
    trainingSetPath = \
        folder / ('TrainingSetWithLocalNormalTraffic.csv' if withLocalNormalTraffic else 'TrainingSet.csv')
    testingSetPath = \
        folder / ('TestingSetWithLocalNormalTraffic.csv' if withLocalNormalTraffic else 'TestingSet.csv')
    trainingSet = pd.read_csv(trainingSetPath)
    testingSet = pd.read_csv(testingSetPath)

    trainingY = trainingSet['Label'].values
    trainingX = \
        trainingSet.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP',
                                  'Dst Port', 'Protocol', 'Timestamp', 'Label']).values
    testingY = testingSet['Label'].values
    testingX = \
        testingSet.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP',
                                 'Dst Port', 'Protocol', 'Timestamp', 'Label']).values

    trainingX = np.nan_to_num(trainingX, nan=0, posinf=0, neginf=0)
    testingX = np.nan_to_num(testingX, nan=0, posinf=0, neginf=0)

    if shuffleDataset:
        trainingX, trainingY = shuffle(trainingX, trainingY, random_state=randomState)
        testingX, testingY = shuffle(testingX, testingY, random_state=randomState)

    return trainingX, testingX, trainingY, testingY


def loadNextCon2020FeatureNames(folder, withLocalNormalTraffic=False):
    folder = Path(folder)
    testingSetPath = \
        folder / ('TestingSetWithLocalNormalTraffic.csv' if withLocalNormalTraffic else 'TestingSet.csv')
    testingSet = pd.read_csv(testingSetPath)
    testingSet = \
        testingSet.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP',
                                 'Dst Port', 'Protocol', 'Timestamp', 'Label'])
    testingSetFeatureNames = testingSet.columns
    return testingSetFeatureNames


def loadSingleFileDataset(filepath, yColName: str, xDropColNames: list,
                          shuffleDataset=False, randomState=None,
                          testSize: Optional[float] = 0.33):
    df = pd.read_csv(str(filepath))
    df = df.dropna()
    y = df[yColName].values
    X = df.drop(columns=xDropColNames).values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    if testSize is None:
        if shuffleDataset:
            X, y = shuffle(X, y, random_state=randomState)
        return X, y
    else:
        return train_test_split(X, y, test_size=testSize,
                                shuffle=shuffleDataset, random_state=randomState)


def loadDoHBrw2020(folder, shuffleDataset=False, randomState=None,
                   testSize: Optional[float] = 0.33):
    """
    Load DoHBrw2020 Dataset

    :param folder: dataset folder
    :param testSize: if None return None-split dataset;
                     otherwise specify testing data size
    :param shuffleDataset: shuffle dataset or not
    :param randomState: random state
    :return: (X, y) if test size is None;
             (trainingX, testingX, trainingY, testingY) if test size is float
    """
    filepath = Path(folder) / 'DoHBrw2020.csv'
    yColName = 'Label'
    xDropColNames = ['SourceIP', 'DestinationIP', 'SourcePort',
                     'DestinationPort', 'TimeStamp', 'Label']
    return loadSingleFileDataset(filepath, yColName, xDropColNames,
                                 shuffleDataset, randomState, testSize)


def loadCicIds2017(folder, shuffleDataset=False, randomState=None,
                   testSize: Optional[float] = 0.33):
    filepath = Path(folder) / 'CICIDS2017.csv'
    yColName = ' Label'
    xDropColNames = [' Destination Port', ' Label']
    return loadSingleFileDataset(filepath, yColName, xDropColNames,
                                 shuffleDataset, randomState, testSize)


def loadNslKdd(folder, shuffleDataset=False, randomState=None):
    def replaceLabel(y):
        labelDict = {
            'DoS': ['apache2', 'back', 'land', 'mailbomb', 'neptune',
                    'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm'],
            'R2L': ['ftp_write', 'guess_passwd', 'imap', 'multihop',
                    'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess',
                    'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop'],
            'U2R': ['buffer_overflow', 'httptunnel', 'loadmodule', 'perl',
                    'ps', 'rootkit', 'sqlattack', 'xterm'],
            'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
            'Normal': ['normal']
        }
        for cat, subcatList in labelDict.items():
            for subcat in subcatList:
                y[y == subcat] = cat
        return y

    xEncoder = OrdinalEncoder()
    folder = Path(folder)
    trainingSetPath = folder / 'KDDTrain+.csv'
    testingSetPath = folder / 'KDDTest+.csv'
    trainingSet = pd.read_csv(str(trainingSetPath))
    testingSet = pd.read_csv(str(testingSetPath))

    trainingY = trainingSet['class'].values
    trainingX = trainingSet.drop(columns=['class', 'difficulty'])
    testingY = testingSet['class'].values
    testingX = testingSet.drop(columns=['class', 'difficulty'])

    trainingY = replaceLabel(trainingY)
    testingY = replaceLabel(testingY)

    xEncoder.fit(pd.concat([trainingX[['protocol_type', 'service', 'flag']],
                            testingX[['protocol_type', 'service', 'flag']]],
                           ignore_index=True))
    trainingX[['protocol_type', 'service', 'flag']] = \
        xEncoder.transform(trainingX[['protocol_type', 'service', 'flag']])
    trainingX = trainingX.values
    testingX[['protocol_type', 'service', 'flag']] = \
        xEncoder.transform(testingX[['protocol_type', 'service', 'flag']])
    testingX = testingX.values

    if shuffleDataset:
        trainingX, trainingY = shuffle(trainingX, trainingY, random_state=randomState)
        testingX, testingY = shuffle(testingX, testingY, random_state=randomState)

    return trainingX, testingX, trainingY, testingY


def loadUnswNb15(folder, shuffleDataset=False, randomState=None):
    xEncoder = OrdinalEncoder()
    folder = Path(folder)
    trainingSetPath = folder / 'UNSW_NB15_training-set.csv'
    testingSetPath = folder / 'UNSW_NB15_testing-set.csv'
    trainingSet = pd.read_csv(str(trainingSetPath))
    testingSet = pd.read_csv(str(testingSetPath))

    trainingY = trainingSet['attack_cat'].values
    trainingX = trainingSet.drop(columns=['id', 'attack_cat', 'label'])
    testingY = testingSet['attack_cat'].values
    testingX = testingSet.drop(columns=['id', 'attack_cat', 'label'])

    xEncoder.fit(pd.concat([trainingX[['proto', 'service', 'state']],
                            testingX[['proto', 'service', 'state']]],
                           ignore_index=True))
    trainingX[['proto', 'service', 'state']] = \
        xEncoder.transform(trainingX[['proto', 'service', 'state']])
    trainingX = trainingX.values
    testingX[['proto', 'service', 'state']] = \
        xEncoder.transform(testingX[['proto', 'service', 'state']])
    testingX = testingX.values

    if shuffleDataset:
        trainingX, trainingY = shuffle(trainingX, trainingY, random_state=randomState)
        testingX, testingY = shuffle(testingX, testingY, random_state=randomState)

    return trainingX, testingX, trainingY, testingY


def loadBankChurners(folder, shuffleDataset=False, randomState=None, testSize: Optional[float] = 0.33):
    filepath = Path(folder) / 'BankChurners.csv'
    yColName = 'Attrition_Flag'
    xDropColNames = ['CLIENTNUM',
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_'
                     'Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_'
                     'Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                     'Attrition_Flag']
    return loadSingleFileDataset(filepath, yColName, xDropColNames,
                                 shuffleDataset, randomState, testSize)


def loadDataset(name, folder, shuffleDataset=False, randomState=None, **kwargs):
    datasetDict = {
        'NextCon2020': loadNextCon2020,
        'DoHBrw2020': loadDoHBrw2020,
        'CicIds2017': loadCicIds2017,
        'NslKdd': loadNslKdd,
        'UnswNb15': loadUnswNb15,

        'BankChurners': loadBankChurners,
    }
    return datasetDict[name](folder, shuffleDataset, randomState, **kwargs)


def ordinalEncode(trX, teX=None, numFirst=True, returnEncoder=False):
    _, dim = trX.shape
    if numFirst:
        encodedMask = list()
        for index in range(dim):
            try:
                trX[:, index] = trX[:, index].astype(float)
            except ValueError:
                encodedMask.append(index)
        encodedMask = np.array([(index in encodedMask) for index in range(dim)])
    else:
        encodedMask = np.ones(dim, dtype=bool)

    encoder = OrdinalEncoder().fit(trX[:, encodedMask])
    trX[:, encodedMask] = encoder.transform(trX[:, encodedMask])

    if teX is not None:
        teX[:, ~encodedMask] = teX[:, ~encodedMask].astype(float)
        teX[:, encodedMask] = encoder.transform(teX[:, encodedMask])

    retTuple = (trX,)
    if teX is not None:
        retTuple = tupleAppend(retTuple, teX)
    if returnEncoder:
        retTuple = tupleAppend(retTuple, returnEncoder)
    return retTuple
