#!/usr/bin/env python3
"""Build an SAE cache with feature selection fitted only on training records."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent; SOURCE=ROOT.parent/'scripts'/'prefill-mvp';sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(ROOT))
import train_gru_prefill as base

DEFAULT_DATASET=ROOT/'runs'/'1500'/'dataset.final.jsonl'
DEFAULT_OUT=ROOT/'cache'/'sae500_1500_train_selected'
base.PER_LAYER_TOPK=50;base.FEATURES_PER_TOKEN=500;base.WINDOW=1;base.INPUT_SIZE=500

def split_indices(record_count: int, seed: int = 42):
    order=np.random.RandomState(seed).permutation(record_count);train_end=int(.70*record_count);validation_end=train_end+int(.15*record_count)
    return order[:train_end],order[train_end:validation_end],order[validation_end:]

def response_topic_labels_only(examples,tokenizer,labels):
    marker=tokenizer("<start_of_turn>model\n",add_special_tokens=False)["input_ids"]
    cleaned=[]
    for example,values in zip(examples,labels):
        values=np.asarray(values).copy();token_ids=example["input_ids"]
        matches=[i for i in range(len(token_ids)-len(marker)+1) if token_ids[i:i+len(marker)]==marker]
        if not matches: raise ValueError("assistant marker not found")
        values[:matches[-1]+len(marker)]=base.PAD_LABEL
        values[values==base.LABEL2ID[base.NEUTRAL_LABEL]]=base.PAD_LABEL
        cleaned.append(values)
    return cleaned

def build_features(saes,hidden,train_indices):
    lengths=[array.shape[0] for array in hidden]; columns=[];selected={}
    for layer_offset,layer in enumerate(base.LAYERS):
        train_x=torch.from_numpy(np.concatenate([hidden[i][:,layer_offset,:] for i in train_indices],0)).cuda().float()
        sums=torch.zeros(saes[layer].d_sae,device='cuda')
        for start in range(0,len(train_x),base.SAE_CHUNK_TOKENS): sums+=base.sae_encode(saes[layer],train_x[start:start+base.SAE_CHUNK_TOKENS]).sum(0)
        indices=sums.topk(base.PER_LAYER_TOPK).indices.sort().values;selected[layer]=indices.cpu().numpy();del train_x,sums
        all_x=torch.from_numpy(np.concatenate([array[:,layer_offset,:] for array in hidden],0)).cuda().float()
        encoded=[base.sae_encode(saes[layer],all_x[start:start+base.SAE_CHUNK_TOKENS])[:,indices] for start in range(0,len(all_x),base.SAE_CHUNK_TOKENS)]
        columns.append(torch.cat(encoded).cpu().numpy());del all_x,encoded;torch.cuda.empty_cache();print(f'layer {layer} complete',flush=True)
    flat=np.concatenate(columns,axis=1).astype(np.float16); sequences=[];offset=0
    for length in lengths: sequences.append(flat[offset:offset+length]);offset+=length
    return sequences,selected

def main(dataset: Path = DEFAULT_DATASET, output: Path = DEFAULT_OUT):
    base.DATASET_PATH=str(dataset)
    output.mkdir(parents=True,exist_ok=True);tok,model,saes=base.load_model_and_saes();examples=base.load_and_parse_dataset(tok)
    train,val,test=split_indices(len(examples));hidden,labels=base.extract_hidden_states(tok,model,examples);del model;torch.cuda.empty_cache()
    labels=response_topic_labels_only(examples,tok,labels);features,selected=build_features(saes,hidden,train)
    lengths=np.asarray([len(x) for x in features],dtype=np.int32)
    np.save(output/'features.npy',np.concatenate(features));np.save(output/'labels.npy',np.concatenate(labels).astype(np.int16));np.save(output/'lengths.npy',lengths)
    np.savez(output/'selected_features.npz',**{f'layer_{k}':v for k,v in selected.items()});np.savez(output/'split_indices.npz',train=train,validation=val,test=test)
    (output/'metadata.json').write_text(json.dumps({'dataset':str(dataset.resolve()),'feature_selection':'training records only','seed':42,'split':'70/15/15','records':len(examples),'train':len(train),'validation':len(val),'test':len(test)},indent=2)+'\n')
    print(f'wrote {output}',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();main(args.dataset,args.output)
