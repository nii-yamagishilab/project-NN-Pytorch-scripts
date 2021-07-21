# Work flow for model training

The script works in this way during model training. <NNN.MMM> are files in each project directory.

Please view this Markdown file using a wide screen.

```sh
     <main.py>       Entry point and controller of training process
        |           
   Argument parse    core_scripts/config_parse/arg_parse.py
        |             Parse the input argument. 
        |             You can add new arguments to this file
        |
Initialize random    core_scripts/startup_config.py
seed, torch.backend   Set random seed for torch and Numpy
        |             Set torch.backends.cudnn.deterministic
        |             Set torch.backends.cudnn.benchmark
        |
   Choose device     
        |            Only single GPU is currently supported
        | 
Initialize & load 
training data set
        |----------|
        .     Load data set <config.py> 
        .     configuration 
        .          |        
        .          |
        .     Loop over     core_scripts/data_io/customize_dataset.py
        .     data subset
        .          |       
        .          |---------|
        .          .    Load one    core_scripts/data_io/default_data_io.py
        .          .    subset
        .          .         |
        .          .    Validate configuration
        .          .    in <config.py>
        .          .         |
        .          .         |
        .          .    Is data duration 
        .          .    information available
        .          .    in <DATASET_NAME.dic>?
        .          .         |              \
        .          .         | yes           \ no
        .          .    Load                  Scan data and save duration 
        .          .    <DATASET_NAME.dic>    to <DATASET_NAME.dic>
        .          .         |               /
        .          .         |              /
        .          .    Is data mean/std 
        .          .    information available
        .          .    in <DATASET_NAME.bin>?
        .          .         |              \
        .          .         | yes           \ no
        .          .    Load                  Scan data and compute mean/std
        .          .    <DATASET_NAME.bin>    save to <DATASET_NAME.bin>
        .          .         |               /
        .          .         |              /
        .          .    Wrap data subset around
        .          .    class NIIDataSet(torch.utils.data.Dataset)
        .          .         |
        .          .    Add torch.util.data.DataLoader
        .          .    	 |
        .          .    Wrap DataSet and DataLoader into
        .          .    class NIIDataSetLoader
        .          .         |
        .          |---------|
        .          |
        .     Add DataLoader to read
        .     multiple subsets
        .          |
        .     Combine data subsets into
        .     class NII_MergeDataSetLoader
        .     	   |
        |----------|
        |
Initialize & load 
development data set  
        |
Initialize Model     <model.py>
    Model()           Model is a meta-model that includes
        |             common utilities: 
        |             load_mean_std, normalize_input, forward()
        |             See example in project/*/model.py
        | 
Initialize Loss      <model.py>
    Loss()             Loss() is defined in <model.py>
        |              Loss.forward(A, B) computes training loss
        |              where A will be returned by from Model.forward()
        |              B will be loaded from output data of the dataset
        |              See example in project/01-nsf/hn-nsf/model.py
        |              
        |              If Model.forward() returns the loss directly
        |              You can set Loss.forward(A, B) to return A 
        |              See example in project/05-nn-vocoders/blow/model.py
        |
Initialize Optimizer core_scripts/op_manager/op_manager.py
        |
Load checkpoint as   This is specified by --trained-model option to main.py
a object
        |
        |
Start model training       
        |             
        |            core_scripts/nn_manager/nn_manager.py  
        |             f_train_wrapper()       
        |             This is used most of models I have.
        |             But GANs need nn_manager_GAN.py       
        |----------|
        .     Set training   
        .     config        
        .          |
        .     Set training  core_scripts/op_manager/op_process_monitor.py
        .     monitor        set buffer to store training loss 
        .          |         set buffer to store validation loss
        .          |         track the best epoch
        .          |
        .          |
        .     Checkpoint 
        .     available?
        .          |        \
        .          | No      \ yes
        .          |        Load model weights, optimizer status, training log
        .          |        Training log can be ignored by 
        .          |          --ignore-training-history-in-trained-model
        .          |        Optimizer status can be ignored by
        .          |          --ignore-optimizer-statistics-in-trained-model
        .          |         /
        .          |        /
        .          |
        .     User defined  <model.py>
        .     setup          
        .          |-------|
        .          .   Model.other_setups() is defined?         
        .          .       |                \
        .          .       | No              \ Yes
        .          .       |        Run Model.other_setups()
        .          .       |                 /
        .          .       |                /
        .          .   Model.g_pretrained_model_path and 
        .          .   Model.g_pretrained_model_prefix defined?
        .          .       |                \
        .          .       | No              \ Yes
        .          .       |        Load model parameters available
        .          .       |        core_scripts/nn_manager/nn_manager_tools.py
        .          .       |        f_load_pretrained_model_partially()
        .          .       |        
        .          .       |        This is used to initialize part of the model
        .          .       |        using pre-trained components
        .          .       |                 /
        .          .       |                /
        .          |-------|---------------|
        .          | 
        .          |
        .     Loop over training data
        .     for one epoch
        .          |
        .          |-------|    core_scripts/nn_manager/nn_manager.py
        .          |       |    f_run_one_epoch()
        .          |       |
        .          |  Loop over 
        .          |  training data
        .          |       |
        .          |       |-------|
        .          |       .    get data_in, data_tar, data_info 
        .          |       .    from DataLoader
        .          |       .       |
        .          |       .       
        .          |       .    Put data_in to device
        .          |       .       |
        .          |       .    if Model.forward()
        .          |       .    requires target data  
        .          |       .       |        \
        .          |       .       | No      \ Yes  
        .          |       .       |       Put data_tar to device
        .          |       .       |       This behavior is on by 
        .          |       .       |       main.py --model-forward-with-target 
        .          |       .       |       see example in 
        .          |       .       |       project/05-nn-vocoder/wavenet
        .          |       .       |         /
        .          |       .       |        /
        .          |       .    if Model.forward()
        .          |       .    requires data file 
        .          |       .    information (data name)
        .          |       .       |        \
        .          |       .       | No      \ Yes
        .          |       .       |       Save file info to data_info
        .          |       .       |       This behavior is on by 
        .          |       .       |       main.py --model-forward-with-file-name 
        .          |       .       |       see example in 
        .          |       .       |       project/05-nn-vocoder/blow
        .          |       .       |         /
        .          |       .       |        /
        .          |       .    Call data_gen <- Model.forward(...)   <mode.py>
        .          |       .    with data_in, data_tar, data_info
        .          |       .    
        .          |       .       |       data_gen can be anything.
        .          |       .       |       If can be generated data by model
        .          |       .       |       (e.g., project/01-nsf/hn-nsf)
        .          |       .       |       Or the training loss value itself
        .          |       .       |       (e.g., project/05-nn-vocoder/wavenet)
        .          |       .       | 
        .          |       .    Whether Model.loss()                 <mode.py>
        .          |       .    is defined
        .          |       .       |        \
        .          |       .       | No      \ Yes
        .          |       .       |       Model.loss(data_gen, data_tar)
        .          |       .       |       This is not used anymore
        .          |       .       |         /
        .          |       .       |        /
        .          |       .    Call Loss.compute()      <mode.py>
        .          |       .    with data_gen, data_tar
        .          |       .       |       data_gen can be anything
        .          |       .       |       If data_gen is the already being the
        .          |       .       |       the loss value, Loss.compute()
        .          |       .       |       can simply return data_gen
        .          |       .       |       (e.g., project/05-nn-vocoder/wavenet)
        .          |       .       |
        .          |       .    Save loss
        .          |       .    loss.backward()
        .          |       .    optimizer.step()
        .          |       .       |
        .          |       |-------|
        .          |       |
        .          |       |
        .          |  Log down time passed 
        .          |       |
        .          |  Log down the training
        .          |  loss over one epoch
        .          |       |
        .          |  Loop over development data
        .          |  for one epoch
        .          |       |
        .          |  Log down time passed 
        .          |       |
        .          |  Log down the validation
        .          |  loss over one epoch
        .          |       |
        .          |  Best epoch so far?
        .          |       |       \
        .          |       | No     \ yes
        .          |       |        Save model as 
        .          |       |        trained_model.pt
        .          |       |        /
        .          |       |       /
        .          |  Save current model and optimizer
        .          |  as epoch*.pt
        .          |       |
        .          |  Early stop?
        .          |       | No    \
        .          |       |        \ Yes
        .          |<------|        |
        .                           |
        |---------------------------|
        |
       Done
```

# Work flow for inference
It is similar to training. 

```sh
     <main.py>       Entry point and controller of training process
        |           
   Argument parse    core_scripts/config_parse/arg_parse.py
        |             Parse the input argument. 
        |             You can add new arguments to this file
        |
Initialize random    core_scripts/startup_config.py
seed, torch.backend   Set random seed for torch and Numpy
        |             Set torch.backends.cudnn.deterministic
        |             Set torch.backends.cudnn.benchmark
        |
   Choose device     
        |            Only single GPU is currently supported
        | 
Initialize & load    <config.py> 
training data set
        |
        |
Initialize Model     <model.py>
    Model()           Model is a meta-model that includes
        |             common utilities: 
        |             load_mean_std, normalize_input, forward()
        |             See example in project/*/model.py
        |
Load trained-model   This is specified by --trained-model option to main.py
        |
        |
Start inference
        |            core_scripts/nn_manager/nn_manager.py  
        |             f_inference_wrapper()       
        |             
        |----------|
        .     Loop over inference data
        .          |        
        .          |-------|
        .          |   Model.inference() is defined?         
        .          |       |                \
        .          |       | No              \ Yes
        .          |       |        Use Model.inference()
        .          |   Use Model.forward()   /
        .          |       |                /
        .          |       |---------------|
        .          |       |
        .          |       |
        .          |    get data_in, data_tar, data_info 
        .          |    from DataLoader
        .          |       |
        .          |    Put data_in to device
        .          |       |
        .          |    if Model.forward() or Model.inference()
        .          |    requires target data  
        .          |       |        \
        .          |       | No      \ Yes  
        .          |       |       Put data_tar to device
        .          |       |       This behavior is on by 
        .          |       |       main.py --model-forward-with-target 
        .          |       |       see example in 
        .          |       |       project/05-nn-vocoder/wavenet
        .          |       |         /
        .          |       |        /
        .          |    if Model.forward()
        .          |    requires data file 
        .          |    information (data name)
        .          |       |        \
        .          |       | No      \ Yes
        .          |       |       Save file info to data_info
        .          |       |       This behavior is on by 
        .          |       |       main.py --model-forward-with-file-name 
        .          |       |       see example in 
        .          |       |       project/05-nn-vocoder/blow
        .          |       |         /
        .          |       |        /
        .          |    Call data_gen <- Model.forward(...)   <mode.py>
        .          |    with data_in, data_tar, data_info
        .          |       |
        .          |       |
        .          |    Save data_gen to output file
        .          |       |       data_gen is a concatenated tensor.
        .          |       |       But data_gen will be split according to
        .          |       |        output_dims in <config.py>.
        .          |       |       If output_dims = [10, 1], data_gen[:, 0:10]
        .          |       |        is be saved as one file, data_gen[:, 10:11]
        .          |       |        is another file. 
        .          |       |       File name extension and format is
        .          |       |        decided by output_exts in <config.py>.
        .          |       |       If output_exts = [".mel", ".f0"], 
        .          |       |       the two output files will be "NAME.mel" and 
        .          |       |       "NAME.f0".
        .          |       |       Unless it is "NAME.wav", everthing will be
        .          |       |       saved as binary little-endian float32 format
        .          |<------|
        .          |
        |----------|
        |
       Done
```