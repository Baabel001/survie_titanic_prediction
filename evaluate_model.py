from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def accu_display(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}\n\nClassification Report:\n{classification_report(y, y_pred)}')
    
def confusion_matrix(y, y_pred):
    
    conf_matrix = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
                                 display_labels=["Survived", "Not Survived"])
    disp.plot()