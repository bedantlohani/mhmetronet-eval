import torch

def convert(path_to_old_model, path_to_save_converted_model):
    """
    path_to_old_model is the path to old model
    and
    path_to_save_converted_model is the path where the converted model is stored
    """
    old_wgts = torch.load(path_to_old_model, map_location=lambda storage, loc: storage)
    new_wgts = OrderedDict()
    new_wgts['encoder.weight']=old_wgts['0.encoder.weight']
    new_wgts['encoder_dp.emb.weight']=old_wgts['0.encoder_with_dropout.embed.weight']
    new_wgts['rnns.0.weight_hh_l0_raw']=old_wgts['0.rnns.0.module.weight_hh_l0_raw']
    new_wgts['rnns.0.module.weight_ih_l0']=old_wgts['0.rnns.0.module.weight_ih_l0']
    new_wgts['rnns.0.module.weight_hh_l0']=old_wgts['0.rnns.0.module.weight_hh_l0_raw']
    new_wgts['rnns.0.module.bias_ih_l0']=old_wgts['0.rnns.0.module.bias_ih_l0']
    new_wgts['rnns.0.module.bias_hh_l0']=old_wgts['0.rnns.0.module.bias_hh_l0']
    new_wgts['rnns.1.weight_hh_l0_raw']=old_wgts['0.rnns.1.module.weight_hh_l0_raw']
    new_wgts['rnns.1.module.weight_ih_l0']=old_wgts['0.rnns.1.module.weight_ih_l0']
    new_wgts['rnns.1.module.weight_hh_l0']=old_wgts['0.rnns.1.module.weight_hh_l0_raw']
    new_wgts['rnns.1.module.bias_ih_l0']=old_wgts['0.rnns.1.module.bias_ih_l0']
    new_wgts['rnns.1.module.bias_hh_l0']=old_wgts['0.rnns.1.module.bias_hh_l0']
    new_wgts['rnns.2.weight_hh_l0_raw']=old_wgts['0.rnns.2.module.weight_hh_l0_raw']
    new_wgts['rnns.2.module.weight_ih_l0']=old_wgts['0.rnns.2.module.weight_ih_l0']
    new_wgts['rnns.2.module.weight_hh_l0']=old_wgts['0.rnns.2.module.weight_hh_l0_raw']
    new_wgts['rnns.2.module.bias_ih_l0']=old_wgts['0.rnns.2.module.bias_ih_l0']
    new_wgts['rnns.2.module.bias_hh_l0']=old_wgts['0.rnns.2.module.bias_hh_l0']

    torch.save(new_wgts, path_to_save_converted_model+'converted_model.pth')

if __name__ == '__main__':
    convert("/home/bedant/mh-metronet/mcnn_shtechB_110.h5","/home/bedant/mh-metronet" )
