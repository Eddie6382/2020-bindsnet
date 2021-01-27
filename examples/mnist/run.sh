rm -r plots/performance/*
rm -r plots/weights/*
rm -r plots/assignments/*
rm -r plots/inputs/*
rm -r plots/spikes/*
rm -r plots/voltages/*
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.01
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.05
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.1
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.2
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.3
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.5
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 10 --plot --thres_ratio 0.9

