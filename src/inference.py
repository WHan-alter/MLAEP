from model import covid_prediction_model
import torch
import esm
import numpy as np
import pandas as pd
from dataset import DMS_data
from sklearn.model_selection import KFold
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seq', type=str, default=0,help='csv file of input sequences')
parser.add_argument('out', type=str, default=0,help='Output dir')
parser.add_argument('--prediction', type=bool, default=False,help='whether to output the prediction results')
parser.add_argument('--embeddings', type=bool, default=False,help='whether to output the prediction embeddings')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model=covid_prediction_model(freeze_bert=True)
model.load_state_dict(torch.load('trained_model/pytorch_model.bin',map_location=device)) ## load the trained model 


print("model in device:",next(model.parameters()).device)
model.eval()


data=pd.read_csv(args.seq,index_col=0)
idx,strs,tokens=batch_converter([(1,s) for s in data.seq.to_list()])
all_result=np.empty((0,9,2))
embedding=np.empty((0,9,2580))
for batch_token in torch.split(tokens,10):
    predict=model(batch_token.cuda(),labels=None)
    embedding=np.append(embedding,predict.embedding.cpu().detach().numpy(),axis=0)
    all_result=np.append(all_result,predict.logits.cpu().detach().numpy(),axis=0)
if args.prediction:
    with open(args.out+"/prediction.npy","ab") as f1 :
        np.save(f1,all_result)

if args.embeddings:
    with open(args.out+"/embedding.npy","ab") as f2 :
        np.save(f2,embedding)
