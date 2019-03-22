from keras.models import Sequential, load_model
import h5py
import sys, getopt, re, os
import numpy as np
_convert_map = {
    'Dense'				: _convert_dense,
    'Activation'			: _convert_activation,
    'ReLU'				: _convert_advanced_activation,
    'LeakyReLU'				: _convert_advanced_activation,
    'PReLU'				: _convert_advanced_activation,
    'ELU'				: _convert_advanced_activation,
    'ThresholdedReLU'			: _convert_advanced_activation,
    'Softmax'				: _convert_advanced_activation,

    'AveragePooling2D'			: _convert_pooling,
    'MaxPooling2D'			: _convert_pooling,
    'GlobalAveragePooling2D'		: _convert_pooling,
    'GlobalMaxPooling2D'		: _convert_pooling,
    'Conv2D'				: _convert_convolution,
    'Conv2DTranspose'			: _convert_convolution,
    'DepthwiseConv2D'			: _convert_convolution,
    'SeparableConv2D'			: _convert_separable_convolution,

    'Flatten'				: _convert_flatten,
    'Reshape'				: _convert_reshape,
    'Concatenate'			: _convert_concat,
    'BatchNormalization'		: _convert_batchnorm,

    'InputLayer'			: _default_skip,
    'Dropout'				: _default_skip,
    'SpatialDropout2D'			: _default_skip,
    'SpatialDropout1D'			: _deafult_skip,
}
def main(argv):
    hdf5file = ''
    outputname = ''
    batch_size = 1
    try:
        opts, args = getopt.getopt(argv,"hi:o:b:",["ifile=","ofile=","batch="])
    except getopt.GetoptError:
        print('keras_parser.py -i <hdf5file> -o <outputname> -b <batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <hdf5file> -o <outputname> -b <batch-size>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            hdf5file = arg
        elif opt in ("-o", "--ofile"):
            outputname = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
    layerDict = symparse(hdf5file)
    array_path = weight_gen(outputname, layerDict)
    file_gen(outputname, batch_size, layerDict, array_path)


def _convert_dense(index,keras_layer,layerDict,dtype):
    config = keras_layer.get_config()
    layerDict[index]['type'] = 'Dense'
    layerDict[index]['name'] = config['name']
    if hasattr(keras_layer, ('batch_input_shape')):
        layerDict[index]['input_shape'] = config['batch_input_shape'][1]
    else:
        layerDict[index]['input_shape'] = layerDict[index-1]['output_shape']
    layerDict[index]['output_shape'] = config['units']
    weightList = keras_layer.get_weights()
    layerDict[index]['kernel'] = weightList[0]
    layerDict[index]['bias'] = weightList[1]
    layerDict[index]['act_type'] = config['activation']
    layerDict[index]['dtype'] = dtype
    return layerDict

def _convert_activation(index,keras_layer,layerDict,dtype):
    config =  keras_layer.get_config()
    layerDict[index]['type'] = 'Activation'
    layerDict[index]['name'] = config['name']
    layerDict[index]['input_shape'] = layerDict[index-1]['output_shape']
    layerDict[index]['output_shape'] = layerDict[index-1]['output_shape']
    layerDict[index]['act_type'] = config['activation']
    layerDict[index]['dtype'] = dtype
    return layerDict

def _convert_flatten(index,keras_layer,layerDict,dtype):
    config = keras_layer.get_config()
    layerDict[index]['type'] = 'Flatten'
    layerDict[index]['name'] = config['name']
    layerDict[index]['dtype'] = dtype
    return layerDict

def _convert_reshape(index,keras_layer,layerDict,dtype):
    config = keras_layer.get_config()
    layerDict[index]['type'] = 'Reshape'
    layerDict[index]['name'] = config['name']
    layerDict[index]['input_shape'] = layerDict[index-1]['output_shape']
    layerDict[index]['output_shape'] = config['target_shape']
    layerDict[index]['dtype'] = dtype
    return layerDict

def symparse(inputfile):
    model = load_model(inputfile)
    for i in range(len(model.layers)):
        keras_layer = model.get_layer(index=i)
        config = keras_layer.get_config();
        if i == 0:
            layerDict = [{}]
            dtype = config['dtype']
        else:
            layerDict.append({})
        if keras_layer.name not in _convert_map:
            raise NotImplementedError("{} is not supported".format(keras_layer.name))
        layerDict = _convert_map[type(keras_layer).__name__](i,keras_layer,layerDict,dtype)
#        if 'dense' in keras_layer.name:
#            _convert_dense(i,keras_layer,layerDict,dtype)
#        if 'activation' in keras_layer.name:
#            _convert_activation(i,keras_layer,layerDict,dtype)
    return layerDict

def weight_gen(outputname,layerDict,array_path=[]):
    for i in range(len(layerDict)):
        if layerDict[i]['type'] == 'Dense':
            var = layerDict[i]['name']
            array_path.append('weights/' + var + '_b.npy')
            array_path.append('weights/' + var + '_k.npy')
            np.save('weights/' + var+'_b.npy',layerDict[i]['bias'])
            np.save('weights/' + var+'_k.npy',layerDict[i]['kernel'])
        if layerDict[i]['type'] == 'Activation':
            array_path.append('')
    return array_path

def file_gen(outputname,batch_size,layerDict,array_path):
    name = outputname + '.py'
    f = open(name,'w+')
    f.write('import heterocl as hcl\nimport hlib\nimport numpy as np\n')
    f.write('hcl.init(hcl.Float(32))\n')
    #generate placeholders
    f.write('I = hcl.placeholder(('+ str(batch_size) + ',' + str(layerDict[0]['input_shape']) + '), \"I\")\n')
    layernum = 0
    var_list = ['I']
    for i in range(len(layerDict)):
        if layerDict[i]['type'] == "Dense":
            name_b = 'l' + str(layernum) + '_b'
            var_list.append(name_b)
            name_k = 'l' + str(layernum) + '_k'
            var_list.append(name_k)
            f.write(name_b + ' = hcl.placeholder((' + str(layerDict[i]['output_shape']) + ',), ' + '\"' + name_b + '\")\n')
            f.write(name_k + ' = hcl.placeholder((' + str(layerDict[i]['input_shape']) + ',' + str(layerDict[i]['output_shape']) + '), ' + '\"' + name_k + '\")\n')
        if layerDict[i]['type'] == 'Activation' and i!=(len(layerDict)-1):
            name = 'act_'+str(layernum)
            var_list.append(name)
            f.write(name + ' = hcl.placeholder((' + str(batch_size) +',' + str(layerDict[i]['output_shape']) + '), ' + '\"' + name + '\")\n')
        layernum = layernum + 1
    f.write('O = hcl.placeholder((' + str(batch_size) + ',' + str(layerDict[layernum-1]['output_shape']) + '), ' + '\"O\")\n')
    var_list.append('O')
    #generate function
    f.write('def ' + outputname + '('+ var_list[0])
    for i in range(1,len(var_list)):
        f.write(',' + var_list[i])
    f.write('):\n')
    next_var = 'I'
    layernum = 0
    for i in range(len(layerDict)):
        if layerDict[i]['type'] == 'Dense':
            name = 'l' + str(layernum) + '_out'
            name_k = 'l' + str(layernum) + '_k'
            name_b = 'l' + str(layernum) + '_b'
            f.write('\t' + name + ' = hlib.nn.dense('+ next_var + ',' + name_k + ',' + name_b + ')\n')
            next_var = name
        if layerDict[i]['type'] == 'Activation':
            if i!=(len(layerDict)-1):
                name = 'act_' + str(layernum)
                if layerDict[i]['act_type'] == 'tanh':
                    f.write('\t' + name + ' = hlib.nn.' + layerDict[i]['act_type'] + '(' + next_var + ')\n')
                else:
                    f.write('\thlib.nn.'+ layerDict[i]['act_type'] + '(' + name + ',' + next_var + ')\n')
                next_var = name
            else:
                f.write('\thlib.nn.' + layerDict[i]['act_type'] + '(O,' + next_var + ')\n')
        layernum = layernum + 1
    #generate schedule and build
    f.write('s = hcl.create_schedule([' + var_list[0])
    for i in range(1,len(var_list)):
        f.write(',' + var_list[i])
    f.write('],' + outputname + ')\n')
    f.write('f= hcl.build(s)\n') 
    #generate arrays
    layernum = 0
    input_list = []
    for i in range(len(layerDict)-1):
        layer = layerDict[i]
        if layerDict[i]['type'] == 'Dense':
            name_b = layer['name'] + 'b'
            name_k = layer['name'] + 'k'
            input_list.append(name_b)
            input_list.append(name_k)
            f.write(name_b + " = hcl.asarray(np.load(" + "\'" + array_path[layernum] + "\'))\n")
            f.write(name_k + " = hcl.asarray(np.load(" + "\'" + array_path[layernum+1] + "\'))\n")
            layernum = layernum + 2
        if layerDict[i]['type'] == 'Activation':
            f.write(layer['name'] + " = hcl.asarray(np.zeros("+var_list[layernum+1]+'.shape))\n')
            input_list.append(layer['name'])
            layernum = layernum + 1
    f.write('_in = hcl.asarray(np.random.randint(256,size=I.shape))\n')
    f.write('_out = hcl.asarray(np.zeros(O.shape))\n')
    input_list.append('_out')
    f.write('f(_in')
    for i in range(len(input_list)):
        f.write(','+input_list[i])
    f.write(')\n')
    f.write('print(_out)\n')
    f.close()
if __name__ == "__main__":
    main(sys.argv[1:])

