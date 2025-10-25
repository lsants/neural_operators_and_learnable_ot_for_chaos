This is the implementation for the workshop paper TBD

```
@article{tbd,
  title={TBD},
  author={TBD},
  journal={TBD},
  year={2025}
}
```

### Prepare the training data
To generate the problem data in the `datagen` folder, modify the ```exp_config.json``` file to define the dynamical system and run:
```
python generate_data.py
```


### Optimal transport experiment
To train the emulator with the optimal transport (OT) method for chosen data, modify the config files in ```/config``` and run:
```
bash experiments/new/srun.sh
```

### Evaluation
TBD
