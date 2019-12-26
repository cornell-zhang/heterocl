To properly use this framework, perform the following setup:
# Using Keras
Keras uses two different methodologies to build up a neural network: Sequential and Model. Sequential only requires users to insert the different neural layers back to back, while Model requires users to specify exactly how each layer is connected. The differences are shown below:
```python   
    #sequential Model
    seq_model = Sequential()
    seq_model.add(Dense(32, input_dim=784))
    
    #Normal Model
    a = Input(shape=(32,))
    b = Dense(32)(a)
    mod_model = Model(inputs=a, outputs=b)
    
    #Save the model
    seq_model.save("my_seq_model.h5")
    mod_model.save("my_model.h5")
    
    #Load the model
    model = load_model("my_model.h5")
    s_model = load_model("my_seq_model.h5")
```

To insert a model into the HeteroCL framework, you can use *model* directly. If you want to download the model or reload it, perform the code shown in above in the bottom two lines.
# Using HeteroCL
1. Download and setup HeteroCL and TVM from github
3. Once both python environments from the githubs are set up, go to the HeteroCL github, and from the main directory, go to the "python/" and "hlib/python/" folders and execute the function "python setup.py install --user" in each.
Now that the environment is properly set up, here is how to compile a Keras model into a HeteroCL model.

1. In a Python script, put into the header "from heterocl.frontend import get_relay_model".
2. The function requires the following inputs: (*model*, *shape*, *frontend*, *dtype*, *in_params*.), where *model* is the Keras model, *shape* is the dictionary of inputs, *frontend* is the frontend being used, *dtype* is the data type, and *in_params* is an optional input if the parameters are not included in the model. The function can handle models from two different sources:
    1. If the model was saved and exported from Keras in an HDF5 file, set "model" to the file path to the model.
    2. if the model is created in the Python script, just set "model" to the Keras model output.
For the shape inputs, users have to include the inputs name and shape as the key and value to the shape dictionary input. For other inputs like weights that define the model, those parameters do not need to be created by users as the weights are included in the Keras model. The rest of the inputs can be left blank.
3. the function outputs a HeteroCL function (```func```) and the list of parameters needed for the model (```params```). To insert an image or tensor into the model, create the input and output tensors by putting in the data as a NumPy array. For inputs, set them as ```in_x = hcl.asarray(numpy_array)``` and for outputs set them as ```out_x =   hcl.asarray(np.zeros(out_shape))```. put the inputs and the outputs into their own arrays (eg. ```[in_1,in_2,... in_n]```).
4. execute the function as follows:
```func(*in_array,*params,*out_array)```.
5. The output is placed into out_array and if you want to convert them back into NumPy use the function ```out_array[i].asnumpy()```.