from pickle import FALSE, TRUE
from random import uniform
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.ncde_encoder import  NCDEEncoder
from models.mask import generate_geom_mask
from models.losses_rec import  masked_mae_loss, get_align_loss, get_uniform_loss
from models.losses_cl import contrastive_loss
from models.gen_neg import gen_neg_view
from utils import take_per_row, split_without_nan, centerize_vary_length_series, torch_pad_nan
import math
import tasks

class CTRL:
    '''The CTRL model'''

    def __init__(
        self,
        input_dims,
        output_dims=320,
        intial_dims=128,
        hidden_dims=64,
        depth=4, 
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        lm = 5,
        mask_ratio = 0.5,
        taskW = 0.1,
        maskW = 0.8,
        hard_neg = None,
        debiase = True,
        threshold = 0.98,
        topk = 0.2,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a CTRL model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
    
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = NCDEEncoder(input_dims=input_dims, output_dims=output_dims, intial_dims = intial_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        nParams = sum([p.nelement() for p in self._net.parameters()])
        print('Number of model parameters is', nParams, flush=True)
        print(self._net)
       

        self.net = torch.optim.swa_utils.AveragedModel(self._net) 
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        self.mask_mode = 'geom'#'binomial'#
        self.lm = lm
        self.mask_ratio = mask_ratio
        self.taskW = taskW

        self.maskW = maskW
        self.hard_neg = hard_neg
        self.early_stop =False
        self.debiase = debiase
        self.threshold = threshold
        self.topk = topk
      



    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the CTRL model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3

        
        early_stop_steps = 3
        min_loss = 100000000
        last_loss = 1e8
        best_epoch = 0

        if not self.early_stop:
            if n_iters is None and n_epochs is None:
                if train_data.size > 100000:
                    n_iters = 400
                elif train_data.size > 40000:
                    n_iters = 200
                else:
                    n_iters = 100

        print("n_iters:",n_iters)
        print(" raw train_data:",train_data.shape) #(n_instance, n_timestamps, n_features)
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_without_nan(train_data, sections, axis=1), axis=0) 
           
        print("train_data:",train_data.shape)
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            cum_contrast_loss = 0
            cum_temporal_loss = 0
            cum_instance_loss = 0
            cum_mask_loss1 = 0
            cum_mask_loss2 = 0 

            cum_align_loss = 0
            cum_uniform_loss = 0


            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    print("__break__")
                    break
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                x1 = x
                x2 = x

                optimizer.zero_grad()
                
                #view 1
                b, seq_len1, f = x1.size()
                if self.mask_mode == 'geom':            
                    nan_mask = ~x1.isnan()
                    mask1 = torch.from_numpy(generate_geom_mask(seq_len1, b*f, masking_ratio=self.mask_ratio, lm=self.lm)).to(x1.device)
                    mask1 = mask1.reshape(seq_len1, b, f).transpose(0,1)
                    mask1 |= (~nan_mask) 
                    mask1[:,0,:] = True
                elif self.mask_mode == 'all_true':
                    mask1 = x1.new_full(x1.size(), True, dtype=torch.bool)
                
                z1, out1, _  = self._net(x1.clone(), mask1) 

                masked_loss1 = masked_mae_loss(out1, x1, ~mask1)
                unmasked_loss1 = masked_mae_loss(out1, x1, mask1)
                loss1 = self.maskW * masked_loss1 + (1-self.maskW) * unmasked_loss1
               
                #view 2
                b, seq_len2, f = x2.size()
                if self.mask_mode == 'geom':  
                    nan_mask = ~x2.isnan()       
                    mask2 = torch.from_numpy(generate_geom_mask(seq_len2, b*f, masking_ratio=self.mask_ratio, lm=self.lm)).to(x2.device)
                    mask2 = mask2.reshape(seq_len2, b, f).transpose(0,1)
                    mask2 |= (~nan_mask)
                    mask2[:,0,:] = True
                elif self.mask_mode == 'all_true':
                    mask2 = x2.new_full(x2.size(), True, dtype=torch.bool)

                z2, out2, _ = self._net(x2.clone(), mask2)
            
                masked_loss2 = masked_mae_loss(out2, x2, ~mask2)
                unmasked_loss2 = masked_mae_loss(out2, x2, mask2)
                loss2 = self.maskW * masked_loss2 + (1-self.maskW) * unmasked_loss2
             
                assert z1.size()==z2.size(), f'z1:{z1.size()}  z2:{z2.size()}'
              
                #view hard neg
                if self.hard_neg is not None:
                    neg_x = gen_neg_view(x1.clone(), self.hard_neg)
                    b, seq_len_neg, f = neg_x.size()
                    if self.mask_mode == 'geom':            
                        nan_mask = ~neg_x.isnan()
                        neg_mask = torch.from_numpy(generate_geom_mask(seq_len_neg, b*f, masking_ratio=self.mask_ratio, lm=self.lm)).to(neg_x.device)
                        neg_mask = neg_mask.reshape(seq_len_neg, b, f).transpose(0,1)
                        neg_mask |= (~nan_mask)
                        neg_mask[:,0,:] = True 
                    elif self.mask_mode == 'all_true':
                        neg_mask = neg_x.new_full(neg_x.size(), True, dtype=torch.bool)
                    
                    z_neg, out_neg, _ = self._net(neg_x.clone(), neg_mask)
                else:
                    neg_x = None
                    z_neg = None

                contrast_loss, instance_loss, temporal_loss  = contrastive_loss(
                    z1, 
                    z2, 
                    z_neg,
                    temporal_unit=self.temporal_unit,
                    debiase= self.debiase and ((n_epochs and self.n_epochs>0.1*n_epochs) or (n_iters and self.n_iters > 0.1*n_iters)),
                    threshold = self.threshold,
                    topk = self.topk
                )
                
                enc1 = z1.reshape(b*seq_len1, -1)
                enc1_norm = enc1/torch.norm(enc1, p=2, dim=-1, keepdim=True)
                enc2 = z2.reshape(b*seq_len2, -1)
                enc2_norm = enc2/torch.norm(enc2, p=2, dim=-1, keepdim=True)

                align_loss = get_align_loss(enc1_norm, enc2_norm)
                uniform_loss1 = get_uniform_loss(enc1_norm)
                uniform_loss2 = get_uniform_loss(enc2_norm)

               
                loss = self.taskW * contrast_loss  + (loss1 +loss2)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net) 

                    
                cum_contrast_loss += contrast_loss.item()
                cum_temporal_loss += temporal_loss.item()
                cum_instance_loss += instance_loss.item()
                cum_mask_loss1 += loss1.item()
                cum_mask_loss2 += loss2.item()
                cum_loss += loss.item()

                cum_align_loss += align_loss.item()
                cum_uniform_loss += (uniform_loss1.item()+ uniform_loss2.item())

                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
                
            
            if interrupted:
                print("__break__")
                break
            
            cum_loss /= n_epoch_iters
            cum_contrast_loss /= n_epoch_iters
            cum_instance_loss /= n_epoch_iters
            cum_temporal_loss /= n_epoch_iters
            cum_mask_loss1 /= n_epoch_iters
            cum_mask_loss2 /= n_epoch_iters
            cum_align_loss /= n_epoch_iters
            cum_uniform_loss /= n_epoch_iters
            

            loss_log.append(cum_contrast_loss)
            if verbose:               
                print(f"Epoch #{self.n_epochs}:total_loss={cum_loss}, contrast_loss={cum_contrast_loss}, mask_loss1={cum_mask_loss1}, mask_loss2={cum_mask_loss2}, align_loss={cum_align_loss}, uniform_loss={cum_uniform_loss}")
            self.n_epochs += 1

    
            # print('loss lower:', (last_loss-cum_loss)/last_loss)
            # if  last_loss-cum_loss > 0.01 * last_loss:                             
            #     best_epoch = self.n_epochs
            #     print(f'best epoch {best_epoch} *****************')
            # elif self.early_stop and self.n_epochs - best_epoch > early_stop_steps: 
            #     print('Early stopped.')
            #     break 
            # last_loss = cum_loss
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_contrast_loss)
            
        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out, _ , _ = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()

    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'geom',  'all_true' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy() #[n_TS,n_timestamps,n_features]
    

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def adjust_learning_rate(optimizer, lr, epoch, epochs):
        """Decay the learning rate based on schedule"""
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        print('lr:',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
