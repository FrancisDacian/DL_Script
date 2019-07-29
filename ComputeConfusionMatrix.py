import os
import scipy.io as scio
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) # Not output by scientific notation

gt_root = ''
out_mat_root = ''


#labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
labels = [0, 1, 2, 3, 4, 5]
no_class = 6
con_mat = np.zeros((no_class, no_class))
con_mat_all = np.zeros((no_class, no_class))

def read_mat(gt_root, out_mat_root, mat_name):

    GT_mat = scio.loadmat(gt_root + mat_name)
    OP_mat = scio.loadmat(out_mat_root + mat_name)
    GT_array = GT_mat['GTcls'][0][0][0]
    OP_array = OP_mat['out_mat']
    print(GT_array, np.unique(GT_array), OP_array, np.unique(OP_array))
    width = GT_array.shape[0]
    length = GT_array.shape[1]
    GT_temp = np.zeros((width, length))
    print(GT_temp.shape)
    # replace the 255 in Ground Truth with output mat.
    GT_temp = np.where(GT_array == 255, OP_array, GT_array)
    print(GT_temp)
    return GT_temp, OP_array, width, length

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(out_mat_root + 'confusion_matrix.png')


if __name__ == '__main__':
   for i in range(27):
       mat_name = str(i) + '.mat'
       gt_data, output_data, wi, le = read_mat(gt_root, out_mat_root, mat_name)
       for j in range(wi):
           con_mat_temp = confusion_matrix(gt_data[j], output_data[j], labels)
           print(con_mat_temp)
           con_mat += con_mat_temp
   con_mat_all += con_mat
   plt.figure()
   plot_confusion_matrix(con_mat_all, classes=labels, normalize=True, title='Normalized confusion matrix')
