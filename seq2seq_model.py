
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from tqdm import trange
from torchcrf import CRF
import pandas as pd
import csv
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from get_embedding import get_embedding
from tensorboardX import SummaryWriter
from keras.utils import to_categorical
tokenizer = AutoTokenizer.from_pretrained("/home/zengjiaqi/Bio_Clinical_Bert")

#writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
sequence_length = 100
input_size = 768
hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 32
num_epochs = 10
learning_rate = 0.001
max_length=100
zero = [0 for i in range(768)]#零向量 768维

def construct_dataset():
    x_train_np = np.load('./seq2seq/train_x3.npy',mmap_mode='r')
    y_train_np = np.load('./seq2seq/y_train3.npy',mmap_mode='r')
    #y_train_np = to_categorical(y_train_np)

    x_train = torch.from_numpy(x_train_np[200:960]).float()
    y_train = torch.from_numpy(y_train_np[200:960])
    x_valid = torch.from_numpy(x_train_np[0:200]).float()
    y_valid = torch.from_numpy(y_train_np[0:200])

    t=pd.read_csv('selected_all_sentence.csv',header=None)
    #t = pd.read_csv('./seq2seq/has_label2.csv',header=None)
    texts=[]
    for i in range(20000,21000):
        texts.append(t.loc[i][1])
    x_test_np,_,_ = get_embedding('sentence',want_to_write_actual_textcsv=False,texts=texts,save_array=False)
    x_test = torch.from_numpy(x_test_np).float()
    #y_test = torch.from_numpy(y_train_np)


    train_dataset = TensorDataset(x_train,y_train)
    validation_dataset = TensorDataset(x_valid,y_valid)
    test_dataset = TensorDataset(x_test,x_test)#临时凑数
    print(type(test_dataset))
    #Data Loader
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size = batch_size, shuffle=False)
    return train_loader,validation_loader,test_loader

#BIRNN
class bilstm_crf(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(bilstm_crf, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size,hidden_size,num_layers,batch_first = True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
        self.crf = CRF(num_classes,batch_first=True)

    def forward(self, x):
        #set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        #out, _ = self.lstm(x,(h0,c0))#out:tensor of shape (batch_size,seq_length,hidden_size*2
        out, _ = self.lstm(x,h0)#out:tensor of shape (batch_size,seq_length,hidden_size*2

        #decode the hidden state of the last time step
        out = self.fc(out)
        #print('out size:',out.size())
        return out

model = bilstm_crf(input_size,hidden_size, num_layers,num_classes).to(device)
#model = torch.load('./seq2seq/seq2seq_model2.pth')
#model = torch.load('./model.pth')
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
train_loader,validation_loader,test_loader = construct_dataset()

def validation(epoch):
    #test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_loader:
            images = images.reshape(-1,sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            #outputs = model.crf.decode(outputs)
            outputs = torch.argmax(outputs,dim=2)
            #labels = labels.numpy().tolist()
            total += images.size(0)*images.size(1)
            correct += (outputs==labels).sum().item()
            #for i in range(len(labels)):
             #   if (outputs[i]==labels[i]):
              #      correct+=1
            #print(total,correct)

        print('Test Accuracy of the model on the validation sentences: {} %'.format(100 * correct / total))
        #writer.add_scalar('validation accuracy',100 * correct / total,epoch)

def train():
    #train the model
    total_step = len(train_loader)
    print('total_step:',total_step)

    for epoch in range(num_epochs):
        res = 0

        for i, (sentence,labels) in enumerate(train_loader):
            sentence = sentence.reshape(-1,sequence_length, input_size).to(device)
            labels = labels.to(device).long()
            labels = torch.reshape(labels,(-1,))
            #print(train_loader,sentence.size(),labels.size())
            #forward pass
            y_hat = model(sentence)
            y_hat = torch.reshape(y_hat,(-1,3))
            #print('y_hat size:',y_hat.size())
            #loss = -model.crf(y_hat, labels)

            loss = criterion(y_hat,labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%2 == 0:
                print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs, i+1,total_step, loss.item()))
        #writer.add_scalar('training loss',res,epoch)
        validation(epoch)

    #save the model checkpoint

    torch.save(model,'./seq2seq/seq2seq_model_nocrf.pth')

def test(test_file):
    output = []

    with torch.no_grad():

        for sentence,_ in test_loader:
            print('sentence size:',sentence.size())
            sentence = sentence.reshape(-1,sequence_length, input_size).to(device)
            y_hat = model(sentence)
            #y_hat = model.crf.decode(y_hat)#(batch_size,sequence_length)
            y_hat = torch.argmax(y_hat,dim=2)
            print('y_hat size:',y_hat.size())
            y_hat = y_hat.cpu().numpy().tolist()
            for i in range (len(y_hat)):
                output.append(y_hat[i])
    output=np.asarray(output)

    print(output.shape) #output为每句话的[0,0,1,2,2,2,2,0,0,……]
    #print(output[0:100])
    result = [] #result为每句话中事件的起止位置如[2,6]
    for i in range(output.shape[0]):
        result_i=[]
        for j in range(output.shape[1]-1):
            if (j==0 and output[i][j]!=0):
                result_i.append(j) #对于开头为1或2
            if (output[i][j]==0 and output[i][j+1]!=0):
                result_i.append(j+1)
            if (output[i][j]!=0 and output[i][j+1]==0):
                result_i.append(j)
            if (j==output.shape[1]-2 and output[i][j+1]!=0):
                result_i.append(j+1) #对于结尾为……012
        result.append(result_i)
    #print(result)

    print(result)
    test_csv = pd.read_csv(test_file,header=None)
    f=open('seq2seq_test_nocrf.csv','w')
    #f=open('seq2seq_train_performance.csv','w')
    csv_write = csv.writer(f)
    csv_write.writerow(['file_name','sentence','disease'])
    for i in trange(len(result)):
        print(result[i])
        if(len(result[i])==0):
            row=[]
            for j in range(2):
                row.append(test_csv.loc[i+20000][j])
            csv_write.writerow(row)
        else:
            row=[]
            for j in range(2):
                row.append(test_csv.loc[i+20000][j])
            #for j in range(0,len(result[i]),2):
             #   row.append(result[i][j]+1)#+1对应人手工标注
              #  row.append(result[i][j+1]+1)

            for j in range(0,len(result[i]),2):
                tokenized_text = tokenizer.tokenize(test_csv.loc[i+20000][1]) #用tokenizer对句子分词
                print('len tokenized_text: ',len(tokenized_text))
                start = result[i][j]
                end = result[i][j+1]+1############???????????????不造为啥报错
                #length = 0
                d=''
                for k in range(start,end):
                    if(end<len(tokenized_text)):#############
                        #print(tokenized_text[k])
                        #length+=len(tokenized_text[k])
                        tokenized_text[k]=tokenized_text[k].replace('#','')
                        d+=tokenized_text[k]

                #row.append(tokenized_text[start:end])
                row.append(d)
                '''
                search_string=''
                for k in range(start,end):
                    search_string+=tokenized_text[k]
                    if k!=end-1:
                        search_string+='.*?'
                
                all = re.findall(search_string,test_csv.loc[i][1][start:],re.IGNORECASE)
                for s in all:
                    if len(s)>length+5:#xiaxie
                        ss_all = re.findall(search_string,s,re.IGNORECASE)
                        for ss in ss_all:
                            if len(ss)<=length+5:
                                row.apend(ss)
                    else:
                        row.append(s)
                '''


            csv_write.writerow(row)
    f.close()

def main():
    train()
    test('./selected_all_sentence.csv')
    #test('./seq2seq/has_label2.csv')

if __name__ == "__main__":
    main()
