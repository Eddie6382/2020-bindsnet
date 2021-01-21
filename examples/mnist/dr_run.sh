rm -r plots/performance/*
rm -r plots/weights/*
rm -r plots/assignments/*
rm -r plots/inputs/*
rm -r plots/spikes/*
rm -r plots/voltages/*
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --model_path  "400neurons_0.5dr_rate/model.pkl" --dr_rate 0.5
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --model_path  "400neurons_0.4dr_rate/model.pkl" --dr_rate 0.4
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --model_path  "400neurons_0.3dr_rate/model.pkl" --dr_rate 0.3
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --model_path  "400neurons_0.2dr_rate/model.pkl" --dr_rate 0.2
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --model_path  "400neurons_0.1dr_rate/model.pkl" --dr_rate 0.1