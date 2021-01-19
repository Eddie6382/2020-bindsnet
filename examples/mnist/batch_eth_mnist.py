import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.learning.learning import WeightDependentPostPre
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=512)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--clamp", type=float, default=0)
parser.add_argument("--unclamp", type=float, default=0)
parser.add_argument("--dr_rate", type=float, default=0)
parser.add_argument("--txt", type=int, default=0)
parser.add_argument("--model_path", type=str, default=None)
parser.set_defaults(plot=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu
clamp = args.clamp
unclamp = args.unclamp
dr_rate = args.dr_rate
txt = args.txt
model_path = args.model_path

update_interval = update_steps * batch_size

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

'''
===========================================================
TRAINING
===========================================================
'''
    # Build network.
    
if model_path == None:
    network = DiehlAndCook2015(
        n_inpt=784,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        norm=78.4,
        nu=(1e-4, 1e-2),
        theta_plus=theta_plus,
        inpt_shape=(1, 28, 28),
    )

    # Directs network to GPU
    if gpu:
        network.to("cuda")

    # Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)

    # Masking
    mask_dict = dict()
    if clamp > 0:
        mask = np.sort(np.random.choice(n_neurons, int(n_neurons * clamp), replace=False))
        mask_dict["clamp"] = torch.from_numpy(mask)
        print("Always fired neurons:", mask)
        c_mask = np.sort(np.setdiff1d(np.arange(n_neurons), mask))
    elif unclamp > 0:
        mask = np.sort(np.random.choice(n_neurons, int(n_neurons * unclamp), replace=False))
        mask_dict["unclamp"] = torch.from_numpy(mask)
        print("Unfired neurons:", mask)
        c_mask = np.sort(np.setdiff1d(np.arange(n_neurons), mask))
    else:
        mask = np.arange(0)
        c_mask = np.arange(n_neurons)

    # Sequence of accuracy estimates.
    accuracy = {"all": [], "proportion": []}

    # Voltage recording for excitatory and inhibitory layers.
    exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=int(time / dt))
    inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=int(time / dt))
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt)
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt)
        )
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None

    spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

    inc = int(n_train/batch_size)
    print("\nBegin training.\n")
    start = t()

    for epoch in range(n_epochs):
        labels = []

        if epoch % progress_interval == 0:
            print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
            start = t()

        # Create a dataloader to iterate and batch data
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=gpu,
        )

        for step, batch in enumerate(tqdm(train_dataloader)):
            if step > n_train:
                break
            # Get next input sample.
            inputs = {"X": batch["encoded_image"]}
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            if step % update_steps == 0 and step > 0:
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels, device=device)

                # Get network predictions.
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )

                # Compute network accuracy according to available classification strategies.
                accuracy["all"].append(
                    100
                    * torch.sum(label_tensor.long() == all_activity_pred).item()
                    / len(label_tensor)
                )
                accuracy["proportion"].append(
                    100
                    * torch.sum(label_tensor.long() == proportion_pred).item()
                    / len(label_tensor)
                )

                print(
                    "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                    % (
                        accuracy["all"][-1],
                        np.mean(accuracy["all"]),
                        np.max(accuracy["all"]),
                    )
                )
                print(
                    "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                    " (best)\n"
                    % (
                        accuracy["proportion"][-1],
                        np.mean(accuracy["proportion"]),
                        np.max(accuracy["proportion"]),
                    )
                )

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            # mask on connections
            mask_e = torch.zeros(n_neurons, device=device)
            mask_np = np.random.choice(a=n_neurons, size=int(n_neurons*dr_rate),replace = False)
            mask_e[torch.from_numpy(mask_np)] = 1
            mask_i = mask_e.expand(784,n_neurons)
            mask_e = torch.diag(mask_e)
            # print(mask_e.bool().shape)
            masks = {("Ae","Ai"):mask_e.bool()}

            labels.extend(batch["label"].tolist())

            # Run the network on the input.
            # network.run(inputs=inputs, time=time, input_time_dim=1, neuron_fault=mask_dict, name={"Ae"}, isReturnSpike=True)
            network.run(inputs=inputs, time=time, input_time_dim=1, dr_mask=masks, train = True,update_rule=WeightDependentPostPre,one_step=True)
            if (not step % update_steps) and (step > 0) and (mask_dict != dict()):
                print("input shape:", inputs["X"].shape)
                print("spike_record shape:", spike_record.shape)

            # Add to spikes recording.
            s = spikes["Ae"].get("s").permute((1, 0, 2))
            spike_record[
                (step * batch_size)
                % update_interval : (step * batch_size % update_interval)
                + s.size(0)
            ] = s

            # Get voltage recording.
            exc_voltages = exc_voltage_monitor.get("v")
            inh_voltages = inh_voltage_monitor.get("v")

            # Optionally plot various simulation information.
            if plot and step > 0 and step % update_steps == 0:
                image = batch["image"][:, 0].view(28, 28)
                inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
                input_exc_weights = network.connections[("X", "Ae")].w
                square_weights = get_square_weights(
                    input_exc_weights.view(784, n_neurons), n_sqrt, 28
                )
                square_assignments = get_square_assignments(assignments, n_sqrt)
                spikes_ = {
                    layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
                }
                voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

                inpt_axes, inpt_ims = plot_input(
                    image, inpt, label=labels[step % update_steps], axes=inpt_axes, ims=inpt_ims
                )
                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                weights_im = plot_weights(square_weights, im=weights_im, save="plots/weights/weights{}.png".format(step + epoch*inc))
                assigns_im = plot_assignments(square_assignments, im=assigns_im, save="plots/assignments/assign{}.png".format(step + epoch*inc))
                perf_ax = plot_performance(accuracy, ax=perf_ax, save="plots/performance/train_acc.png", period=batch_size*update_steps, interval=20)
                voltage_ims, voltage_axes = plot_voltages(
                    voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line", save="plots/voltages/vol.png",
                )

                plt.pause(1e-8)

            network.reset_state_variables()  # Reset state variables.

    print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
    print("Training complete.\n")


'''
===========================================================
TESTING
===========================================================
'''

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

# To see the neuron is correctly clamped

pbar = tqdm(total=n_test)
for step, batch in enumerate(tqdm(test_dataloader)):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1,one_step = True,neuron_fault=mask_dict, name={"Ae"},train = False,dr = dr_rate,isReturnSpike=True)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    if (not step % 1000) and (step > 0) and (mask_dict != dict()):
        print("input shape:", inputs["X"].shape)
        print("spike_record shape:", spike_record.shape)
        print("abnormal neuron's spikes:", torch.count_nonzero(spike_record[:, :, mask]))
        print("normal neuron's spikes:", torch.count_nonzero(spike_record[:, :, c_mask]))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.4f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.4f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
