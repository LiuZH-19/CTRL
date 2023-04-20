import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean(),
        'RSE': rse_np(pred, target),
        'CORR': node_pcc_np(pred, target)
    }

def rse_np(preds, labels):
    #preds [n_TS, n_samples, pred_lens, F]
    mse = np.sum(np.square(np.subtract(preds, labels)).astype('float64')) #
    means = np.mean(labels)#, axis=-2
    labels_mse = np.sum(np.square(np.subtract(labels, means)).astype('float64'))#
    return np.float(np.sqrt(mse/labels_mse))#np.mean()

def node_pcc_np(x, y):
    #preds [n_TS, n_samples, pred_lens, F]
    x = x.swapaxes(0,-2)
    y = y.swapaxes(0,-2)
    #preds [pred_lens, n_samples, n_TS, F]
    H, N, S, F = x.shape
    x = x.reshape(H*N, -1)
    y = y.reshape(H*N, -1)
    print("corr x:",x.shape)
    sigma_x = x.std(axis=0)
    sigma_y = y.std(axis=0)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    corr = ((x - mean_x) * (y - mean_y)).mean(0) / (sigma_x * sigma_y + 0.000000000001)
    return np.float(corr.mean())



    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding):    
    t = time.time()
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )# [n_TS,n_timestamps,n_features]

    infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}

    all_mse = []
    all_mae = []
    all_rse = []
    all_corr = []
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
       
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)#[n_TS,n_samples,pred_len,F] 
        test_labels = test_labels.reshape(ori_shape)
       

        if test_data.shape[0] > 1:
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)
            
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        norm_metrics = cal_metrics(test_pred, test_labels)
        raw_metrics = cal_metrics(test_pred_inv, test_labels_inv)
        all_mse.append(norm_metrics['MSE'])
        all_mae.append(norm_metrics['MAE'])
        all_rse.append(raw_metrics['RSE'])
        all_corr.append(raw_metrics['CORR'])       


        ours_result[pred_len] = {
            'norm': norm_metrics,
            'raw': raw_metrics
        }

    avg = {'all':{}, 'avg_all':{}}   

    avg['avg_all']['MSE'] = np.mean(all_mse)
    avg['avg_all']['MAE'] = np.mean(all_mae)
    avg['avg_all']['RSE'] = np.mean(all_rse)
    avg['avg_all']['CORR'] = np.mean(all_corr)
    
    avg['all']['MSE'] = all_mse
    avg['all']['MAE'] = all_mae
    avg['all']['RSE'] = all_rse
    avg['all']['CORR'] = all_corr


    eval_res = {
        'average':avg,
        'ours': ours_result,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
