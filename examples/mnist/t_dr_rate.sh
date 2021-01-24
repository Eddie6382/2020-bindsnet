rm -r plots/performance/*
rm -r plots/weights/*
rm -r plots/assignments/*
rm -r plots/inputs/*
rm -r plots/spikes/*
rm -r plots/voltages/*
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0   --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.1 --model_path  "400neurons_0.3dr_rate/model.pkl" 
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.3 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.4 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.5 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.6 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.7 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.8 --model_path  "400neurons_0.3dr_rate/model.pkl"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.9 --model_path  "400neurons_0.3dr_rate/model.pkl"
