python train_mul_times.py exchange-rate  ctrl --loader  forecast_hdf --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg shuffle8 --debiase --threshold 0.98 --topk 0.4
python train_mul_times.py wind  ctrl --loader  forecast_hdf --batch-size 128  --max-threads 8   --iters 400 --eval --taskW 0.1 --hard-neg mix4 --debiase --threshold 0.98 --topk 0.4
python train_mul_times.py WTH  ctrl --loader  forecast_csv --batch-size 128  --max-threads 8   --iters 400 --eval --taskW 0.1 --hard-neg shuffle8 --debiase --threshold 0.98 --topk 0.4
python train_mul_times.py ILI  ctrl --loader  forecast_csv --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 101

