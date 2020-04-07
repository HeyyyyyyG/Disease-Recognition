
#from pytorch_transformers import BertModel, BertConfig, BertTokenizer
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from tqdm import trange,tqdm
import csv

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel


def get_embedding(file_or_sentence,want_to_write_actual_textcsv,save_array,file_name=None,texts=None,array_name=None):

    # OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
    #import logging
    #logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('/home/zengjiaqi/bert-base-uncase')
    #tokenizer = AutoTokenizer.from_pretrained("/home/zengjiaqi/Bio_Clinical_Bert")

    def load_text(file):
        text_file=pd.read_csv(file,encoding='utf8')
        texts = []
        for i in trange(len(text_file)):
            if ('positive' in file and text_file.loc[i][2]!='et'):
                texts.append(text_file.loc[i][1])###############################################attention############################################3
            elif ('negative' in file):
                texts.append(text_file.loc[i][1])
        return texts
    # Tokenize input
    if (file_or_sentence=='file'):
        file = file_name#'negative20000_6.csv'
        texts = load_text(file)
    #texts = load_text('./positive/positive1_20.csv')

    elif (file_or_sentence=='sentence'):
        texts = texts

    #for i in range(len(texts)):
     #   texts[i] = '[CLS] '+texts[i]+' [SEP]'

    tokens, segments = [], []
    actual_text_number = []
    for i,text in enumerate(tqdm(texts)):
        tokenized_text = tokenizer.tokenize(text) #用tokenizer对句子分词
        print(len(tokenized_text))
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)#索引列表
        if (len(indexed_tokens)<=100):
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            actual_text_number.append(i)
        else:
            print('too long',i,len(indexed_tokens))
    actual_text = []
    for i in actual_text_number:
        actual_text.append(texts[i])

    def actual_textcsv(actual_text_number,file):#写过滤后的csv
        f=open(file[:len(file)-4]+'_actual.csv','w',encoding='utf8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['file_path','sentence'])
        fin = pd.read_csv(file)
        for i in actual_text_number:
            newline=[]
            for k in fin.loc[i]:
                newline.append(k)
            csv_writer.writerow(newline)
        f.close()

    if (want_to_write_actual_textcsv):
        actual_textcsv(actual_text_number,file_name)

    print('len tokens',len(tokens))
    print(tokens[0])
    #print(segments)



    max_len = 100 #最大的句子长度
    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        #input_masks[j] += padding

    #print('tokens',tokens)
    #print(segments)
    #segments列表全0，因为只有一个句子1，没有句子2
    #转换成PyTorch tensors
    batch_size=700
    for i in trange(int(len(tokens)/batch_size)+1):
        if (i!=len(tokens)/batch_size):
            tokens_tensor = torch.tensor(tokens[batch_size*i:batch_size*i+batch_size])
            segments_tensors = torch.tensor(segments[batch_size*i:batch_size*i+batch_size])
        else:
            tokens_tensor = torch.tensor(tokens[batch_size*i:])
            segments_tensors = torch.tensor(segments[batch_size*i:])
        print(tokens_tensor.size(),segments_tensors.size())

        #model = AutoModel.from_pretrained("/home/zengjiaqi/Bio_Clinical_Bert")
        model = BertModel.from_pretrained('/home/zengjiaqi/bert-base-uncase')
        model.eval()

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        model.to('cuda')

        # Predict hidden states features for each layer
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers_cls = outputs[0]#[:,0,:]
        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
        cls = encoded_layers_cls.cpu().numpy()
        if i==0:
            array=cls
        else:
            array = np.concatenate((array,cls),axis=0)
    print(array.shape)

    #np.save('negative20000_6.npy',array)
    #np.save('positive1_20.npy',array)
    if save_array:
        np.save(array_name,array)
    return array,actual_text,actual_text_number

def main():
    #get_embedding('file',True,save_array=False,file_name='negative20000_6.csv')
    t=['This study provides valuable insights which can contribute to preparedness planning']
    #get_embedding('sentence',want_to_write_actual_textcsv=False,texts=t,save_array=True,array_name='a')
    #get_embedding('file',want_to_write_actual_textcsv=True,save_array=True,file_name='./positive/positive0_20.csv',array_name='./positive/positive0_20')
    #get_embedding('file',want_to_write_actual_textcsv=False,save_array=True,file_name='./csv_file/new_train/positive.csv',array_name='./csv_file/new_train/positive_bert_uncase')

if __name__ == "__main__":
    main()
