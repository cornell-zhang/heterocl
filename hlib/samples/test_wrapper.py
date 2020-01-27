from hlib.frontend import *
import numpy as np
import heterocl as hcl
import pickle

# testing harness for from_nnvm results
# progress bar since this can take a while

def printProgressBar(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        length=100,
        fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def test_wrapper(
        model,
        ir,
        frontend,
        filename,
        data,
        tags,
        _out_shape,
        input_size,
        batch_size=1,
        target=None):
    # get json and parameters
    # _json,params=load_params(func_name)
    #batch_size = _json["batch_size"]
    # print(batch_size)
    # generate function and parameters
    if(ir == "relay"):
        f, params = relay_parser.get_relay_model(
            model, input_size, frontend=frontend)

    # get mnist dataset from npy file
    #data  = np.load(data_name)
    #tags  = np.load(tags_name)
    l = 0
    corr = 0
    samp_num = 0
    tot = data.shape[0] * batch_size
    in_size = ((-1,), data.shape[1:])
    in_size = [element for tupl in in_size for element in tupl]
    printProgressBar(
        l,
        tot,
        prefix="Progress",
        suffix="Tested",
        decimals=2,
        length=50)
    for i in data:
        _input = hcl.asarray(i)
        _out = hcl.asarray(np.zeros(_out_shape))
        f(_input, *params, _out)
        result = np.argmax(_out.asnumpy(), axis=1)
        for k in range(batch_size):
            if result[k] == tags[l][k]:
                corr += 1
            samp_num += 1
        l += 1
        # update progress bar
        printProgressBar(
            samp_num,
            tot,
            prefix="Progress",
            suffix="Tested",
            decimals=2,
            length=50)
    print("{}% accuracy out of {} samples ".format(
        format(corr / tot * 100, '.4f'), str(tot)))
