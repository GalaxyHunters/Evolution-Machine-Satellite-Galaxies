import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sb
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(predictions, probs, train_predictions, train_probs,y_train,y_test):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, 
                                     [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, 
                                      [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print({metric.capitalize()},'Baseline:',{round(baseline[metric], 2)},'Test: ',{round(results[metric], 2)},'Train:', {round(train_results[metric], 2)})
    
    # Calculate false positive rates and true positive rates
    print y_test
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();


def RFC(X_train, X_test, y_train, y_test,features):
    clf = RandomForestClassifier(n_estimators= 100,max_features=3,bootstrap=True, criterion='gini',verbose=1,n_jobs=-1)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    rf_probs = clf.predict_proba(X_test)[:, 1]
    train_rf_predictions = clf.predict(X_train)
    train_rf_probs = clf.predict_proba(X_train)[:, 1]
    print(metrics.classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes = ['Survived', 'Quenched'],
                          title = 'Quenching Confusion Matrix')
    importants_cond = pd.DataFrame({'feature': features,
                       'importance': clf.feature_importances_}).\
                        sort_values('importance', ascending = False)
    evaluate_model(pred, rf_probs, train_rf_predictions, train_rf_probs,y_train,y_test)

    # Calculate roc auc
    roc_value = roc_auc_score(y_test, rf_probs)
    roc_value
    # Display
    importants_cond

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18


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

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)







def SVM_Classification(sgal_SVM_features,labels):
    '''
    calculate multi-class classification and return related evaluation metrics
    '''
    num = np.random.randint(50)
    #num=6
    X_train, X_test, y_train, y_test = train_test_split(sgal_SVM_features.reset_index().drop(['tgid'],axis=1),
                                         labels, 
                                         stratify = labels,
                                         test_size = 0.27, 
                                         random_state = num)
    
    #svc = svm.SVC(C=3, kernel='linear',gamma='auto')
    #svc = svm.SVC(C=66, kernel='poly',degree=2)
    svc = svm.SVC(C=50,kernel='poly',degree=3,gamma='scale',probability=True,class_weight='balanced')

    clf = svc.fit(X_train, y_train) #svm
    y_pred = svc.predict(X_test)
    # array = svc.coef_
    # print(array
    print('Actual Feature Set')
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred)),'\n'
    
    pca = PCA(n_components=2).fit(X_train)
    pca_2d = pca.transform(X_train)
    pca2 = PCA(n_components=2).fit(X_test)
    pca2_2d = pca.transform(X_test)
   
    #svmClassifier_2d =   svm.SVC(C=5,kernel='linear').fit(pca_2d, y_train)
    #svmClassifier_2d =   svm.SVC(C=66,kernel='sigmoid').fit(pca_2d, y_train)
    svmClassifier_2d =   svm.SVC(C=60,kernel='poly',degree=3,gamma='auto',probability=True,class_weight='balanced',shrinking=False).fit(pca_2d, y_train)

    y_pred_2 = svmClassifier_2d.predict(pca2_2d)
    print('Decomposed 2d Feature Set')
    print(metrics.confusion_matrix(y_test,y_pred_2))
    print(metrics.classification_report(y_test,y_pred_2))
    for i in range(0, pca_2d.shape[0]):
        if y_train[i] == 3:
            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=30,marker='^')
        elif y_train[i] == 1:
            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=30,marker='o')
        elif y_train[i] == 4:
            c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    s=30,marker='*')
        elif y_train[i] == 0:
            c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    s=30,marker='^')
    #pl.legend([c1, c2,c3], ['strange', 'Quenched','Survived'])
    #pl.legend([c1, c2,c3,c4], ['strange', 'Quenched','Survived','Dead'])
    x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
    y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
    Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
    Z = Z.reshape(xx.shape)
    pl.contour(xx, yy, Z)
    pl.title('SVM Decision Surface')
    #pl.axis('off')
    pl.legend([c2, c3], ['Quenched', 'Survived'])
    pl.show()
    print(num)
    