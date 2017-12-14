FGS  = 0
BI   = 1
DF   = 3
CW   = 4
RAND = 5

train: train.py
ifeq ($(DATASET), cifar)
	python3 train.py --dataset cifar --learning_rate 0.1 -n 5 --eta 0.03 --decay_step 50000 --max_steps 200000 --model_dir models/cifar --set 0
else
	python3 train.py --dataset mnist --learning_rate 0.5 -n 5 --eta 0.18 --decay_step 50000 --max_steps 200000 --model_dir models/mnist --set 0
endif

gen: gen_adv.py
ifeq ($(DATASET), mnist)
	python3 gen_adv.py --dataset cifar -n 5 --attack $($(ATTACK)) --epsilon 0.03 --model_dir models/mnist --direct 1
else
	python3 gen_adv.py --dataset mnist -n 5 --attack $($(ATTACK)) --epsilon 0.1 --model_dir models/mnist --direct 1
endif

eval: eval.py
ifeq ($(DATASET), mnist)
	python3 eval.py --dataset cifar -n 5 --model_dir models/cifar -rt 5
else
	python3 eval.py --dataset mnist -n 5 --model_dir models/mnist -rt 2
endif
