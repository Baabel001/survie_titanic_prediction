from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

def accu_display(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}\n\nClassification Report:\n{classification_report(y, y_pred)}')

    
def roc(y, y_pred):
    
    fpr , tpr, thresholds = roc_curve(y, y_pred)

    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr)
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('Receiver Operating Characteristic')
    plt.show()

    print("Area under Roc Curve for SVM: ", "%.3f" % auc(fpr, tpr))
    
    
def conf_mat(y, y_pred):
    
    conf_matrix = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
                                 display_labels=["Survived", "Not Survived"])
    disp.plot()