#ettm1
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py ETTm1  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio} --runs 5 --iters 200
done

#ettm2
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py ETTm2  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio}  --runs 5 --iters 200
done

#etth1
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py ETTh1  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio} --runs 5 --iters 100
done

#etth2
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py ETTh2  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio}  --runs 5 --iters 100
done

#Weather
for ratio in $(seq 0.125 0.125 0.5); do
    echo $ratio
    python train_mul_times.py WTH  ctrl${ratio} --loader  imputation  --batch-size 128  --max-threads 8   --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.98 --topk 0.4 --max-train-length 96 --irregular ${ratio}  --runs 5 --iters 400
done

