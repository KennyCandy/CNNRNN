from sklearn.metrics import confusion_matrix
import pylab as pl
y_test=['business', 'business', 'business', 'business', 'business',
 'business', 'business', 'business', 'business', 'business', 'business',
  'business', 'business', 'business', 'business', 'business', 
  'business', 'business', 'business', 'business']

pred=['health', 'business', 'business', 'business', 'business',
       'business', 'health', 'health', 'business', 'business', 'business',
       'business', 'business', 'business', 'business', 'business',
       'health', 'health', 'business', 'health']

cm = confusion_matrix(y_test, pred, labels = "")
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()