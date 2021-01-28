rm -r plots/performance/*
rm -r plots/weights/*
rm -r plots/assignments/*
rm -r plots/inputs/*
rm -r plots/spikes/*
rm -r plots/voltages/*
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.01
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.05
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.1
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.2
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.3
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.5
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.35 --model_path  "400neurons_0.2dr_rate/model.pkl" --repeat 3 --plot --clamp 0.9
echo "================================================================================"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.01
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.05
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.1
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.2
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.3
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.5
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0.2 --model_path  "400neurons_0.1dr_rate/model.pkl" --repeat 3 --plot --clamp 0.9
echo "================================================================================"
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.01
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.05
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.1
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.2
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.3
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.5
python batch_eth_mnist.py --gpu --n_neurons 400 --n_epochs 3 --dr_rate 0 --model_path  "400neurons/model.pkl" --repeat 3 --plot --clamp 0.9


