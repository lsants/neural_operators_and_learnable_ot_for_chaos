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
To generate the problem data in the `datagen` folder, modify the ```/configs/datagen/datagen_config.json``` file to define the dynamical system and run:
```
python /pipeline/datagen/generate_data.py
```


### Optimal transport experiment
To train the emulator with the optimal transport (OT) method for chosen data, modify the config files in ```/config``` and run:
```
bash experiments/new/srun.sh
```

### Evaluation
To just evaluate a trained model, modify ```pipeline/testing/evaluate.py``` to fetch the correct dataset and checkpoint path (In the ```if __name__ == '__main__'``` block) and from the repo folder, run ```python -m pipeline.testing.evaluate```
