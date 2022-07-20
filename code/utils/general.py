import os
from glob import glob
import torch
import pdb
from collections import OrderedDict
#1_18_7-vc_Page_009-4Oq

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs
 
def glob_exrs(path):
    exrs = []
    for ext in ['*.exr']:
        exrs.extend(glob(os.path.join(path, ext)))
    return exrs

def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        # pdb.set_trace()
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if data.get('uv_inp',None) is not None:
            data['uv_inp'] = torch.index_select(model_input['uv_inp'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def split_input_uv(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['wc'] = torch.index_select(model_input['wc'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        split.append(data)
    return split

def split_input_uvinfer(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    # pdb.set_trace()
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels), n_pixels, dim=0)):
        data = model_input.copy()
        data['points'] = torch.index_select(model_input['points'], 1, indx)
        data['norm_points'] = torch.index_select(model_input['norm_points'], 1, indx)
        data['mask'] = torch.index_select(model_input['mask'], 1, indx)
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        split.append(data)
    return split

def split_input_wc(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['wc'] = torch.index_select(model_input['wc'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def split_input_pxc(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['pxc'] = torch.index_select(model_input['pxc'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    # pdb.set_trace()
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def merge_output_batch(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    # pdb.set_trace()
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 2:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                            1).reshape(batch_size, total_pixels)
        else:
            # pdb.set_trace()
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                            1).reshape(batch_size, total_pixels, -1)
            
    return model_outputs


def update_state_dict(state_dict, src_key, dest_key):
    """Converts a state dict keys by replacing src_key with dest key 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace(src_key,dest_key) # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def print_log(expname, epoch, data_index, n_batches, loss_dict, alpha, lrate):
    '''
    takes loss dict and other parameters as input and returns a string to be printed 
    '''
    out=[]
    out.append('{}'.format(expname))
    out.append('[{}]'.format(epoch))
    out.append('({}/{})'.format(data_index, n_batches))
    for k in loss_dict:
        if loss_dict[k] is not None:
            out.append('{}:{:.6f}'.format(k,loss_dict[k].item()))
    out.append('alpha:{}'.format(alpha))
    out.append('lrate:{}'.format(lrate))
    
    out_string= ', '.join(out)
    
    print(out_string)
