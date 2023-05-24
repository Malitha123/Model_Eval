### Configuration conventions

```bash
DATASET
├── MODEL_1
│   ├── EXPERIMENT_A
│   │   └── info_of_variables.yaml
│   └── EXPERIMENT_B
│       └── info_of_variables.yaml
└── MODEL_2
```

For example, for `R2Plus1D` model on `EPIC-KITCHENS`, I name the configs as follows:
```bash
<data_and_inputs>_<backbone>_<pretraining>_<optimization>.yaml
```

For example, among `data_and_inputs`, I have `<spatiotemporal_input_size>` as one attribute.


### Tips

1. Always run an experiment on local scale before letting it run. Check all the config parameters where `# CHECK` has been written. Typically, these are learning rates, batch size, number of frames, spatial size, pretraining model, model/arch, dataset(s).
