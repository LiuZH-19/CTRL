#UEA
python train_mul_times.py EthanolConcentration  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py FaceDetection  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py FingerMovements  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.1 --hard-neg mix2 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py Heartbeat  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix2 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py JapaneseVowels  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py PEMS-SF ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py SelfRegulationSCP1  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py SelfRegulationSCP2  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py SpokenArabicDigits  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 400 --eval --taskW 0.05 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py UWaveGestureLibrary	  ctrl --loader  UEA --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.1 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2

#UCR
python train_mul_times.py Chinatown  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py ECG5000  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg shuffle8 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py ElectricDevices  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 400 --eval --taskW 0.01 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py InsectWingbeatSound  ctrl --loader  UCR --batch-size 64 --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg shuffle8 --debiase --threshold 0.99 --topk 0.2 --max-train-length 256
python train_mul_times.py MelbournePedestrian  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 200 --eval --taskW 0.05 --hard-neg shuffle4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py PowerCons  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg mix4 --debiase --threshold 0.99 --topk 0.2
python train_mul_times.py DodgerLoopDay  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.99 --topk 0.2  --max-train-length 288
python train_mul_times.py DodgerLoopGame  ctrl --loader  UCR --batch-size 128  --max-threads 8   --iters 100 --eval --taskW 0.1 --hard-neg shuffle4 --debiase --threshold 0.99 --topk 0.2  --max-train-length 288



