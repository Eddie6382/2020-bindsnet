We insert fault on neuron by calmp them not to spike during training process and testing process (the same neurons being clamped). 
## Fault on excitory layer
### parameter
unclamp (force to 0) : 0, 40, 80, 120, 160, 200
```python
python batch_eth_mnist.py --gpu --batch_size 32 --update_steps 512 --n_neurons 400 --n_epochs 3 --plot
```
### performance of testing accuracy
| unclamp neurons | all_activity | weight_proportion |
| -------- | -------- | -------- |
| 0     | 87%     | 87%     |
| 40    | 87%     | 88%     |
| 80    | 86%     | 86%     |
| 120   | 84%     | 84%     |
| 160   | 82%     | 82%     |
| 200   | 77%     | 77%     |

```bash
# unclamp rate (10%)
Unfired neurons: [353  27  87 316 386 354 301   0 344 320 285 115  69 323 396 384 246
  53 126 257 120 381 318  84 329  41 269 361 230  86 140 312 282 332  40
 251 169 158 262]
```
### Comparision
- 400 neurons with 200 neurons unfired
    77%, 77%
- 200 neurons
    78%, 79%
    
## Fault on inhibitory accuracy
### parameter
the same as above
### performance of testing accuracy
| unclamp neurons | all_activity | weight_proportion |
| -------- | -------- | -------- |
| 0     | 88%     | 88%     |
| 40    | 86%     | 87%     |
| 80    | 84%     | 84%     |
| 120   | %     | %     |
| 160   | %     | %     |
| 200   | 67%     | 67%     |





