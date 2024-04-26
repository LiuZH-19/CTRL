# CTRL

This repository contains the implementation for the paper An NCDE-based Framework for Universal Representation Learning of Time Series. This is not the final official repository because we want to maintain anonymity.

## Requirements

The recommended requirements for CTRL are specified as follows:
* Python 3.8
* torch==1.8.1
* scipy==1.6.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.2
* statsmodels==0.12.2
* Bottleneck==1.3.2
* torchcde
* h5py

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.

* [UEA datasets](http://www.timeseriesclassification.com) should be put into `datasets/UEA/` so that each data file can be located by `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.

* [Exchange-rate dataset](https://drive.google.com/file/d/1pjgw4vJJffmhDGVzJmpBH5hbKTHBWaqB/view?usp=share_link) (obtained from [LSTNet repository](https://github.com/laiguokun/multivariate-time-series-data)) placed at `datasets/exchange-rate.h5`.

* [Wind dataset](https://drive.google.com/file/d/13L7K6Jkmf-u9--9lDkZQY0JTQIeL5Xqe/view?usp=share_link) (obtained from https://www.kaggle.com/sohier/30-years-of-european-wind-generation.) placed at `datasets/wind.h5`.

* [ILI dataset](https://drive.google.com/drive/folders/1DasX30lzEwcVXYaNeyMlQ0PSmCQSow5h?usp=share_link) (obtained from [Autoformer repository](https://github.com/thuml/Autoformer) ) placed at `datasets/ILI.csv`.

* [Weather dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) (link from [Informer repository](https://github.com/zhouhaoyi/Informer2020)) placed at `datasets/WTH.csv`.

* [ETT dataset](https://github.com/zhouhaoyi/ETDataset)  placed at `datasets/ETT*.csv`.

  


## Usage

To train and evaluate CTRL on a dataset, run the following command:

```train & evaluate
python train_mul_times.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size>  --repr-dims <repr_dims>   --eval --taskW <taskW> --hard-neg <hard_neg> --debiase --threshold <threshold> --topk <topk> --max-train-length <len>  --gpu <gpu> --runs <runs>
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_hdf`, `imputation`. |
| batch_size | The batch size (defaults to 128) |
| repr_dims | The representation dimensions (defaults to 320) |
| eval | Whether to perform evaluation after training                 |
| taskW | The trade-off between contrastive task and reconstruction task. |
| hard-neg | The methods to construct hard negative samples. This can be set to `shuffle4`, `shuffle8`, `mix2`, `mix4` or `None`. |
| debiase | Whether to eliminate false negative samples |
| threshold | The similarity threshold to filter false negatives |
| topk             | Proportion of the topk to screen the false negative samples  |
| max-train-length | For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 201) |
| gpu              | The gpu no. used for training and inference (defaults to 0)  |
| runs             | Number of executions (defaults to 5)                                        |

(For descriptions of more arguments, run `python train_mul_times.py  -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName/RunName_Date_Time/`. 

**Scripts:** To train and evaluate CTRL  on  the datasets included in our paper, you can run the script from the `./scripts` folder or  directly run the python scripts.

For example:

```
python train_mul_times.py PowerCons  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2 --runs 5
```

```
python train_mul_times.py exchange-rate  ctrl --loader  forecast_hdf --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg shuffle8 --debiase --threshold 0.98 --topk 0.4 --runs 5
```

```
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py ETTm1  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio} --runs 5 --iters 200
done
```
