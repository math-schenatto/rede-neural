#
True Positive Rate = True Positives / (True Positives + False Negatives)

False Positive Rate = False Positives / (False Positives + True Negatives)

import matplotlib.pyplot as plt

plt.ylabel('True positive rate')
plt.xlabel('False positive rate')

plt.plot([1,3,2], [1,3,2], label="A")
plt.plot([1,2,3], [2,4,8], label="A")

plt.legend(ncol=3,loc='lower right')
plt.show()
