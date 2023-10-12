# Program3: Variational Autoencoder
## Program Introduction
This program is a simple implementation of Variational Autoencoder (VAE) using Pytorch. 
I implemented four different model structures, including a simple MLP-based VAE, a complicated MLP-based VAE, a CNN-based VAE and a residual-based VAE.
You can adjust the model type, the latent dimension, the learning rate, the depth, hidden size of MLP VAE and many other parameters in the config/config.yaml file.
## Program Structure
The program is structured as follows:
```
- Variational_AutoEncoders/
    - readme.md
    - run.sh
    - train.py  # model training
    - visualization.py  # model evaluation and output visualization
    - checkpoint   # store the model checkpoints
    - config/
        - config.yaml # store the configuration
    - data
        - MNIST
    - dataloader/
        - dataloader.py
    - model/
        - __init__.py
        - VAE_SIMPLE.py  # simple MLP-based VAE
        - MLP_VAE.py     # complicated MLP-based VAE
        - Conv_VAR     # CNN-based VAE
        - Res_VAE.py     # residual-based VAE
    - error  # store the error log when using SJTU HPC
    - output # store the output log when using SJTU HPC
    - logdir # store the tensorboard log
    - plot  # store the visualization results

```

## Run the Program
To run the program, you can adjust the parameters in the config/config.yaml file and run the run.sh file.
Remember that the config file has been set to reproduce the best result in the report.
For example, to run the program in SJTU HPC, you can use the following command:
```bash
sbatch run.sh
```
To run the program in your local machine, you can use the following command:
```bash
python train.py
python visualization.py
```