import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
import time
import datetime
from ctrl import CTRL
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import json
#log
class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def save_checkpoint_callback(
    run_dir,
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def main(args, run_dir, seed):
    print(f'seed {seed}'+'*'*100)
    run_dir = os.path.join(run_dir, str(seed)) 
    os.makedirs(run_dir, exist_ok=True)   
    
    sys.stdout = Logger(os.path.join(run_dir,'log.txt')) 

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=seed, max_threads=args.max_threads)

 
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, args.irregular)        
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, args.irregular)
    
    elif args.loader == 'forecast_hdf':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding = datautils.load_forecast_hdf(args.dataset)
        train_data = data[:, train_slice]
        valid_data = data[:, valid_slice]    
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        valid_data = data[:, valid_slice]    
  
    elif args.loader == 'imputation':
        task_type = 'imputation'
        data, missing_data, missing_mask, train_slice, valid_slice, test_slice, scaler, lens = datautils.load_imputation(args.dataset, args.irregular)
        train_data = missing_data[:, train_slice]
        valid_data = missing_data[:, valid_slice]
                
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

       
        
    if args.irregular > 0:
        if task_type not in ['classification', 'imputation']:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        lm = args.lm,
        mask_ratio = args.mask_ratio,
        taskW = args.taskW,
        maskW = args.maskW,
        hard_neg = args.hard_neg,
        debiase = args.debiase,
        threshold = args.threshold,
        topk = args.topk
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(run_dir, args.save_every, unit)

    
    
    t = time.time()
    
    model = CTRL(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')


    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding)
        elif task_type == 'imputation':
            out, eval_res = tasks.eval_imputation(model, data, missing_data, missing_mask, train_slice, valid_slice, test_slice, scaler, lens)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        print('Evaluation result:', eval_res)

        tf = open(f'{run_dir}/eval_res.json', "w")
        json.dump(eval_res,tf)
        tf.close()

        sys.stdout.flush()
        return task_type, eval_res
    print("Finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=201, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 201)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--lm', type=int, default = 5, help='The average length of continuous masking')
    parser.add_argument('--mask-ratio', type=float, default=0.5, help='The mask ratio')
    parser.add_argument('--taskW', type=float, default=0.1, help='The trade-off between contrastive task and reconstruction task')
    parser.add_argument('--maskW',type = float, default = 0.8, help='The trade-off between masked loss and unmasked loss in reconstruction task')
    parser.add_argument('--hard-neg', type=str, default= None, help='The methods to construct hard negative samples: shuffle4, shuffle8, mix2, mix4')
    parser.add_argument('--debiase', action="store_true", help='Whether to eliminate false negative samples')
    parser.add_argument('--threshold', type=float, default=0.99, help='The similarity threshold to filter false negatives')
    parser.add_argument('--topk', type=float, default =0.2, help='Proportion of the topk to screen the false negative samples')
    parser.add_argument('--runs', type=int, default =5, help='Number of executions')
    
    args = parser.parse_args()
            
    run_dir = os.path.join('training',args.dataset, name_with_datetime(args.run_name))
    os.makedirs(run_dir, exist_ok=True)

    all_res = []

    for seed in range(args.runs):
        task_type, eval_res = main(args, run_dir, seed)
        if task_type == 'classification' :
            all_res.append([[eval_res['acc'], eval_res['auprc']]])            
        elif task_type == 'forecasting':
            seed_res = []
            for tmp in ['MSE', 'MAE', 'RSE', 'CORR']:
                seed_res += eval_res['average']['all'][tmp] 
            seed_res +=[eval_res['average'][avg][tmp] for avg in ['avg_all'] for tmp in ['MSE', 'MAE', 'RSE', 'CORR']]  
            all_res.append([seed_res])      
        elif task_type == 'imputation':
             all_res.append([[eval_res['norm']['MSE'],eval_res['norm']['MAE'], eval_res['raw']['MSE'], eval_res['raw']['MAE']]])
        else:
            assert False
    
    all_res = np.concatenate(all_res, axis = 0)
    print('all_res.shape:', all_res.shape)
    if len(all_res.shape)<3:
        np.savetxt(f'{run_dir}/all_res.csv', all_res, delimiter=",", fmt='%.5f')
    mean_res = np.round(all_res.mean(0),5)
    std_res = np.round(all_res.std(0),5)
    print('result mean:', mean_res)
    print('result std:', std_res)

    np.savetxt(f'{run_dir}/mean_res.txt', mean_res, fmt='%.5f')
    np.savetxt(f'{run_dir}/std_res.txt', std_res, fmt='%.5f')

    
