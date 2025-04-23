# run collaborative filtering (variants) algorithm

import torch

from utils import fill_mask, load_dataset, split_dataset

from collabfilter import CF

__author__ = "Yifeng Tao"

PATH_PREFIX = "/content/drive/MyDrive/CS598_DL4H_Project/CADRE-master/"

args = {}
args['is_train'] = True

args['input_dir'] = PATH_PREFIX+"data/input"
args['output_dir'] = PATH_PREFIX+"data/output"
args['repository'] = "gdsc"
args['drug_id'] = -1
args['use_cuda'] = True
args['use_relu'] = False
#args['init_gene_emb'] = True

args['omic'] = "exp"

#args['use_attention'] = True
#args['use_cntx_attn'] = True

args['embedding_dim'] = 200
args['attention_size'] = 128
args['attention_head'] = 8
args['hidden_dim_enc'] = 200
args['use_hid_lyr'] = False

args['max_iter'] = 1025 #int(384000)
# args['dropout_rate'] = 0.6

# args['learning_rate'] = 0.3
args['weight_decay'] = 3e-4
args['batch_size'] = 8
args['test_batch_size'] = 8
args['test_inc_size'] = 128

args['model_label'] = "CF"

args['use_cuda'] = args['use_cuda'] and torch.cuda.is_available()

print("Loading drug dataset...")
dataset, ptw_ids = load_dataset(input_dir=args['input_dir'], repository=args['repository'], drug_id=args['drug_id'])

train_set, test_set = split_dataset(dataset, ratio=0.8)

# replace tgt in train_set
train_set['tgt'], train_set['msk'] = fill_mask(train_set['tgt'], train_set['msk'])

args['exp_size'] = dataset['exp_bin'].shape[1]
args['mut_size'] = dataset['mut_bin'].shape[1]
args['cnv_size'] = dataset['cnv_bin'].shape[1]

if args['omic'] == 'exp':
  args['omc_size'] = args['exp_size']
elif args['omic'] == 'mut':
  args['omc_size'] = args['mut_size']
elif args['omic'] == 'cnv':
  args['omc_size'] = args['cnv_size']

args['drg_size'] = dataset['tgt'].shape[1]
args['train_size'] = len(train_set['tmr'])
args['test_size'] = len(test_set['tmr'])


print("Hyperparameters:")
print(args)

if __name__ == "__main__":

  # Running Experiments to find the best hyperparameters
  args['test_gene_mask'] = False # Ablation Variable 1
  args['mask_num_genes'] = 100   # Ablation Variable 2
  for i in range(4):
    for dr in [0.4, 0.6, 0.8]:
      for lr in [0.1, 0.3, 0.5]:
        if i == 0:
          print("Running for Collaborative Filtering with dropout:", dr, "and learning rate:", lr)
          args['use_attention'] = False
          args['use_cntx_attn'] = False
          args['init_gene_emb'] = True # only relevant for CADRE
        elif i == 1:
          print("Running for SADRE with dropout:", dr, "and learning rate:", lr)
          args['use_attention'] = False
          args['use_cntx_attn'] = True
          args['init_gene_emb'] = True # only relevant for CADRE
        elif i == 2:
          print("Running for CADRE w/o pretrain with dropout:", dr, "and learning rate:", lr)
          args['use_attention'] = True
          args['use_cntx_attn'] = True
          args['init_gene_emb'] = False
        else:
          print("Running for CADRE with pretrain with dropout:", dr, "and learning rate:", lr)
          args['use_attention'] = True
          args['use_cntx_attn'] = True
          args['init_gene_emb'] = True

        args['dropout_rate'] = dr
        args['learning_rate'] = lr

        model = CF(args)

        model.build(ptw_ids)

        if args['use_cuda']:
          model = model.cuda()


        logs = {'args':args, 'iter':[],
                'precision':[], 'recall':[],
                'f1score':[], 'accuracy':[], 'auc':[],
                'precision_train':[], 'recall_train':[],
                'f1score_train':[], 'accuracy_train':[], 'auc_train':[],
                'loss':[], 'ptw_ids':ptw_ids}

        if args['is_train']:
          print("Training...")
          logs = model.train(train_set, test_set,
              batch_size=args['batch_size'],
              test_batch_size=args['test_batch_size'],
              max_iter=args['max_iter'],
              test_inc_size=args['test_inc_size'],
              logs=logs)

          labels, msks, preds, tmr, amtr = model.test(test_set, test_batch_size=args['test_batch_size'])
          labels_train, msks_train, preds_train, tmr_train, amtr_train = model.test_train(train_set, test_batch_size=args['test_batch_size'])

          logs["preds"] = preds
          logs["msks"] = msks
          logs["labels"] = labels
          logs['tmr'] = tmr
          logs['amtr'] = amtr

          logs['preds_train'] = preds_train
          logs['msks_train'] = msks_train
          logs['labels_train'] = labels_train
          logs['tmr_train'] = tmr_train
          logs['amtr_train'] = amtr_train

        else:
          print("LR finding...")
          logs = model.find_lr(train_set, test_set,
              batch_size=args['batch_size'],
              test_batch_size=args['test_batch_size'],
              max_iter=args['max_iter'],
              test_inc_size=args['test_inc_size'],
              logs=logs)
          
  # Testing Ablation and Extensions
  args['test_gene_mask'] = True  # True or False whether to test masking of genes
  args['mask_num_genes'] = 100   # Number of genes to mask
  dr = 0.4  # Best dropout rate from experiments
  lr = 0.5  # Best learning rate from experiments
  for i in range(4):
    if i == 0:
      print("Running for Collaborative Filtering with dropout:", dr, "and learning rate:", lr)
      args['use_attention'] = False
      args['use_cntx_attn'] = False
      args['init_gene_emb'] = True # only relevant for CADRE
    elif i == 1:
      print("Running for SADRE with dropout:", dr, "and learning rate:", lr)
      args['use_attention'] = False
      args['use_cntx_attn'] = True
      args['init_gene_emb'] = True # only relevant for CADRE
    elif i == 2:
      print("Running for CADRE w/o pretrain with dropout:", dr, "and learning rate:", lr)
      args['use_attention'] = True
      args['use_cntx_attn'] = True
      args['init_gene_emb'] = False
    else:
      print("Running for CADRE with pretrain with dropout:", dr, "and learning rate:", lr)
      args['use_attention'] = True
      args['use_cntx_attn'] = True
      args['init_gene_emb'] = True

    args['dropout_rate'] = dr  
    args['learning_rate'] = lr 

    model = CF(args)

    model.build(ptw_ids)

    if args['use_cuda']:
      model = model.cuda()


    logs = {'args':args, 'iter':[],
            'precision':[], 'recall':[],
            'f1score':[], 'accuracy':[], 'auc':[],
            'precision_train':[], 'recall_train':[],
            'f1score_train':[], 'accuracy_train':[], 'auc_train':[],
            'loss':[], 'ptw_ids':ptw_ids}

    if args['is_train']:
      print("Training...")
      logs = model.train(train_set, test_set,
          batch_size=args['batch_size'],
          test_batch_size=args['test_batch_size'],
          max_iter=args['max_iter'],
          test_inc_size=args['test_inc_size'],
          logs=logs)

      labels, msks, preds, tmr, amtr = model.test(test_set, test_batch_size=args['test_batch_size'])
      labels_train, msks_train, preds_train, tmr_train, amtr_train = model.test_train(train_set, test_batch_size=args['test_batch_size'])

      logs["preds"] = preds
      logs["msks"] = msks
      logs["labels"] = labels
      logs['tmr'] = tmr
      logs['amtr'] = amtr

      logs['preds_train'] = preds_train
      logs['msks_train'] = msks_train
      logs['labels_train'] = labels_train
      logs['tmr_train'] = tmr_train
      logs['amtr_train'] = amtr_train

    else:
      print("LR finding...")
      logs = model.find_lr(train_set, test_set,
          batch_size=args['batch_size'],
          test_batch_size=args['test_batch_size'],
          max_iter=args['max_iter'],
          test_inc_size=args['test_inc_size'],
          logs=logs)