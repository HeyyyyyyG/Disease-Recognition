import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

pos = np.load('./csv_file/new_train/positive_bert_uncase.npy',mmap_mode='r')
neg = np.load('./csv_file/new_train/negative_bert_uncase.npy',mmap_mode='r')
x = np.concatenate((pos,neg[:15000]))
y = [1 for i in range (pos.shape[0])]+[0 for i in range (neg[:15000].shape[0])]
y = np.asarray(y)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.1,shuffle = True)
test_y=test_y.tolist()
print(sum(test_y),len(test_y))
clf = SVC(kernel='rbf',class_weight='balanced',verbose=True,max_iter=-1)
print('begin training')
clf.fit(train_x,train_y)
print('begin prediction')
pred_y = clf.predict(test_x)
print(pred_y)
print(classification_report(test_y,pred_y,digits=4))
