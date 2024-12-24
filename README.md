## Our model
### Preprocess data

Before starting the training, make sure the following folders and files are set up:
- **100k**: Directory containing the dataset files. You should have the following files:
  - `test.pkl`
  - `train.pkl`
  - `validate.pkl`
- **230k.json**: file of vocab json mapping

### Train model
Run all **train_model_100k.ipynb**. The model will be saved as model_checkpoint.pth of every epoch.

## Luo's & Luo's modified model
### Preprocess data
Before starting the training, make sure the following folders and files are set up:

- **ckpts**: Directory to store the model checkpoints for every epoch.
- **data**: Directory containing the dataset files. You should have the following files:
  - `test.pkl`
  - `train.pkl`
  - `validate.pkl`
  - `vocab.pkl`
  
### Train model
Execute `train.py`.  
The following parameters can be adjusted:

- **emb_dim**: (embedding size)  
  Default: `80`  
  Defines the size of the embedding layer in the model.

- **dec_rnn_h**: (The hidden state of the decoder RNN)  
  Default: `512`  
  Specifies the size of the hidden state of the decoder RNN.

- **data_path**: (The dataset's directory)  
  Default: `"./data/"`  
  Path to the directory containing the dataset.

- **add_position_features**: (Use position embeddings or not)  
  Default: `True`  
  Whether or not to use position embeddings in the model.

- **max_len**: (Max size of formula)  
  Default: `150`  
  Specifies the maximum length for formula sequences.

- **dropout**: (Dropout probability)  
  Default: `0.0`  
  The dropout probability used in regularization during training.

- **cuda**: (Use CUDA or not)  
  Default: `True`  
  Whether to use CUDA (GPU) acceleration or not.

- **batch_size**: (Batch size)  
  Default: `32`  
  Defines the batch size during training.

- **epoches**: (Number of epochs)  
  Default: `50`  
  The number of epochs for training the model.

- **lr**: (Learning rate)  
  Default: `3e-4`  
  The initial learning rate used in the optimizer.

- **min_lr**: (Minimum learning rate)  
  Default: `3e-5`  
  The minimum value the learning rate can decay to.

- **sample_method**: (Sampling method for training)  
  Default: `teacher_forcing`  
  The method used for scheduling sampling during training. Choices: `teacher_forcing`, `exp`, `inv_sigmoid`.

- **decay_k**: (Exponential decay base for sampling schedule)  
  Default: `1.0`  
  The base of exponential decay for scheduled sampling, or a constant in inverse sigmoid decay.

- **lr_decay**: (Learning rate decay rate)  
  Default: `0.5`  
  The rate at which the learning rate is decayed.

- **lr_patience**: (Learning rate decay patience)  
  Default: `3`  
  The number of epochs with no improvement before the learning rate is decayed.

- **clip**: (Max gradient norm)  
  Default: `2.0`  
  The maximum gradient norm used for gradient clipping to prevent exploding gradients.

- **save_dir**: (Checkpoint saving directory)  
  Default: `"./ckpts"`  
  The directory where model checkpoints will be saved.

- **print_freq**: (Frequency of printing training messages)  
  Default: `100`  
  The frequency at which training messages (e.g., loss) will be printed.

- **seed**: (Random seed for reproducibility)  
  Default: `2020`  
  The random seed used to ensure reproducible results.

- **from_check_point**: (Train from checkpoint or not)  
  Default: `True`  
  Whether to resume training from a checkpoint or start from scratch.
