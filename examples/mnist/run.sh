rm -r plots/performance/*
rm -r plots/weights/*
rm -r plots/assignments/*
rm -r plots/inputs/*
rm -r plots/spikes/*
rm -r plots/voltages/*
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --unclamp 0.5
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --unclamp 0.4
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --unclamp 0.3
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --unclamp 0.2
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --unclamp 0.1
