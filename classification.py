import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from get_embedding import get_embedding
import pandas as pd
from tqdm import trange
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 100
input_size = 768
hidden_size = 128
num_layers = 3

batch_size = 128
num_epochs = 100
learning_rate = 0.0001
max_length=100


def construct_dataset():
    #positive = np.load('./positive/positive0_20.npy',mmap_mode='r')
    #negative = np.load('./negative/negative20000_6.npy',mmap_mode = 'r')

    positive = np.load('./csv_file/new_train/positive_bert_uncase.npy',mmap_mode='r')
    negative = np.load('./csv_file/new_train/negative_bert_uncase.npy',mmap_mode = 'r')

    x_train = np.concatenate((positive[:12000],negative[:12000]))
    x_train = torch.from_numpy(x_train).float()
    y_train = [1 for i in range(12000)]+[0 for i in range(12000)]
    y_train = torch.Tensor(y_train)

    x_valid = np.concatenate((positive[12000:13000],negative[12000:13000]))
    x_valid = torch.from_numpy(x_valid).float()
    y_valid = [1 for i in range(1000)]+[0 for i in range(1000)]
    y_valid = torch.Tensor(y_valid)

    x_test = np.concatenate((positive[13000:15000],negative[13000:15000]))
    x_test = torch.from_numpy(x_test).float()
    y_test = [1 for i in range(len(positive[13000:15000]))]+[0 for i in range(len(negative[13000:15000]))]
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(x_train,y_train)
    validation_dataset = TensorDataset(x_valid,y_valid)
    test_dataset = TensorDataset(x_test,y_test)#临时凑数
    #print(type(test_dataset))
    #Data Loader
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size = batch_size, shuffle=False)
    return train_loader,validation_loader,test_loader



def validation(loader):
    #test the model
    all_label = []
    all_result = []
    with torch.no_grad():
        correct = 0
        total = 0
        for step,(sentences, labels) in enumerate(loader):
            #print(sentences.size())
            sentences = sentences.to(device)
            labels = labels.to(device)
            outputs = model(sentences)
            outputs = outputs.squeeze(dim=-1)  # (batch_size, )
            result = []
            for i in range(outputs.size(0)):
                if (outputs[i].item()>0.5):
                    result.append(1)
                else:
                    result.append(0)

            label_list = labels.cpu().numpy().tolist()
            #for i in range(len(label_list)):
                #if (label_list[i]!=result[i]):
                    #print('predict fail:',step*batch_size+i)
            all_label = all_label+label_list
            all_result = all_result+result

            result = torch.Tensor(result).to(device)
            total += sentences.size(0)
            correct+=(labels==result).sum().item()
            #print(sentences.size(0),(labels==result).sum().item())
            #print(total,correct)

        print('Test Accuracy of the model on the validation/test sentences: {} %'.format(100 * correct / total))
    acc_precision_recall_f1(all_label,all_result)

def train():
    #train the model
    total_step = len(train_loader)
    print('total_step:',total_step)
    for epoch in range(num_epochs):
        for i, (sentence,labels) in enumerate(train_loader):
            #sentence = sentence.reshape(-1,sequence_length, input_size).to(device)
            sentence = sentence.to(device)
            labels = labels.to(device)
            #print(train_loader,sentence.size(),labels.size())
            #forward pass
            out = model(sentence)  # (16, 1)
            out = out.squeeze(dim=-1)  # (16, )
            #print(out)
            #print(labels)
            loss = criterion(out, labels)
            #print(loss)
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs, i+1,total_step, loss.item()))
        validation(validation_loader)

    #save the model checkpoint

    torch.save(model,'./csv_file/new_train/model_bert_uncase.pth')

def test(test_loader):
    print('test')
    validation(test_loader)

def acc_precision_recall_f1(actual,predicted):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(actual, predicted)
    pre, rec, f1, sup = precision_recall_fscore_support(actual, predicted)
    print("precision:", pre, "\nrecall:", rec, "\nf1-score:", f1, "\naccuracy:", acc)

def fast_test(file_or_sentence,source_file_name,texts=None,file=None):
    if file_or_sentence=='sentence':
        array,actual_texts,actual_text_number= get_embedding('sentence',want_to_write_actual_textcsv=False,texts=texts,save_array=False)
    elif file_or_sentence=='file':
        array= get_embedding('file',file_name=file,want_to_write_actual_textcsv=True,save_array=True,array_name=file)
    x = torch.from_numpy(array)
    with torch.no_grad():
        x = x.to(device)
        out = model(x)
        out = out.squeeze(dim=-1)
        result = []
        for i in range(out.size(0)):
            if (out[i].item()>0.5):
                result.append(1)
            else:
                result.append(0)
        #print(sum(result),len(result))
    f=open('selected_all_sentence.csv','a+',newline='')
    csvwriter = csv.writer(f)
    #csvwriter.writerow(['sentence'])
    for i in range(len(result)):
        if(result[i]==1):
            #delete cls sep
            t=actual_texts[i]
            t=t[t.index('[CLS]')+6:t.index('[SEP]')-1]
            if('(Fig' in t[len(t)-15:]):
                t=t[:t.index('(Fig')]
            if ('bioRxiv preprint' in t):
                t=t[t.index('bioRxiv preprint')+len('bioRxiv preprint '):]
            if ('medRxiv preprint' in t):
                t=t[t.index('medRxiv preprint')+len('medRxiv preprint '):]
            if (t[0:3]==': "'):
                t=t[3:]
            csvwriter.writerow([source_file_name[actual_text_number[i]],t])
    f.close()

    '''    
    if(file_or_sentence=='file'):
        if('positive' in file):
            label=[1 for i in range(len(result))]
        else:
            label = [0 for i in range(len(result))]
        acc_precision_recall_f1(label,result)
    '''

def fast_test_array():
    array = np.load('./positive/positive1+2_20.csv.npy',mmap_mode='r')
    x = torch.from_numpy(array)
    with torch.no_grad():
        x = x.to(device)
        out = model(x)
        out = out.squeeze(dim=-1)
        result = []
        for i in range(out.size(0)):
            if (out[i].item()>0.5):
                result.append(1)
            else:
                result.append(0)
        print(sum(result),len(result))



def fast_select_sentence(file_name):
    file=pd.read_csv(file_name)
    texts=[]
    for i in trange(100000):
        texts.append(file.loc[i][1])
    array,actual_texts,_= get_embedding('sentence',want_to_write_actual_textcsv=True,texts=texts,save_array=True,array_name='100000',file_name=file_name)

    x = torch.from_numpy(array)
    with torch.no_grad():
        x = x.to(device)
        out = model(x)
        out = out.squeeze(dim=-1)
        result = []
        for i in range(out.size(0)):
            if (out[i].item()>0.5):
                result.append(1)
            else:
                result.append(0)
        print(sum(result),len(result))

    f=open('selected.csv','w',newline='')
    csvwriter = csv.writer(f)
    csvwriter.writerow(['sentence'])
    for i in trange(len(result)):
        if(result[i]==1):
            csvwriter.writerow([actual_texts[i]])



model = nn.Sequential(
        nn.Linear(768, 128),
        nn.LeakyReLU(),
        nn.Linear(128,1),
        nn.Sigmoid()
)

model = torch.load('./csv_file/new_train/model_precision.pth')
#model = torch.load('./csv_file/new_train/model.pth')
#model = torch.load('./model.pth')
model.to(device)
criterion = nn.BCELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
train_loader,validation_loader,test_loader = construct_dataset()

print('begin')
train()
test(test_loader)

#fast_test(file_or_sentence='file',file='./negative/negative_modify.csv')
#fast_test(file_or_sentence='sentence',texts=['cite_spans','','text',':'])
#fast_test_array()
#fast_select_sentence('./csv_file/all_sentence.csv')
