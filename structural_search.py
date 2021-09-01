
import logging

import torch
from numpy.linalg import LinAlgError

from pseudopruner.utils import count_flops, infer_masks, \
    get_ready_to_prune, mark_to_prune, make_pruning_effective
from pseudopruner.channel_pruner import ConstChannelPruner, \
    CompensationAwarePruner
from pseudopruner.channel_pruner.comp_aware_pruner import rank_channels
from pseudopruner.compensation import update_weight, register_statistics_hook


# a param-free function to return a Module instance
_make_model = None
# a module implemented
# test(model:Module, ratio:float, batch_size:int, device:str)->float
_test_env = None
# a module implemented
# iter_train(model:Module, ratio:float, batch_size:int, device:str)->None
_train_env = None
# path to the pretrained module, match _make_model
_pretrained_weight_path = ''

# device for all computations
# e.g. 'cuda:0'
_device = None
# used in FLOPs counting and inferring masks
# e.g. (1, 3, 224, 224)
_dummy_input_shape = (1, 3, 32, 32)

# ratio of training data for statistics
# e.g. 0.5 (within (0, 1] )
_stat_ratio = 0.2
# batch size for statistics
_stat_batch_size = 128
# sampling stride on feature maps
_sampling_stride = 1

# a function to device prune or not prune a layer
_layer_filter = None

# sparsity sesarch
_sparsity_search_step = None
_prec_end = None  # lowest precision in tolerance
_prec_start = None  # precision of the pretrained model
_prec_step = None  # skip
_max_sparsity = None  # (0, 1]

# test
_test_batch_size = 512
_test_ratio = 1.0


def bottomup_search():
    # load pre-trained model
    model = _make_model()
    weights = torch.load(_pretrained_weight_path, map_location='cpu')
    model.load_state_dict(weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(_device)

    # compute FLOPs before pruning
    dummy_input = torch.rand(_dummy_input_shape, device=_device)
    full_flops = count_flops(model, dummy_input)

    # set up pruners
    const_pruner = ConstChannelPruner()
    cap_pruner = CompensationAwarePruner()
    get_ready_to_prune(model)

    # set up the layers to be pruned
    module_list = list(model.named_modules())
    module_list = [
        (name, module) for name, module in module_list
        if _layer_filter(name, module)
    ]

    # get feature statistics(mean and covariance)
    logging.info('get statistics')
    stat_handles = []
    for _, module in module_list:
        stat_handle = register_statistics_hook(
            module, sampling_stride=_sampling_stride)

    with torch.no_grad():
        _ = _train_env.iter_train(
            model=model, ratio=_stat_ratio,
            batch_size=_stat_batch_size, device=_device)

    for stat_handle in stat_handles:
        stat_handle.remove()

    # set step constraint for search
    prec_thre = _prec_start
    _prec_step = (_prec_start - _prec_end)/len(module_list)

    # run compensation-aware pruning as channel importance ranking
    for name, module in module_list:
        logging.info(f'rank channels of {name}')
        torch.cuda.empty_cache()
        channel_ranks = rank_channels(module)
        mark_to_prune(module, {
            'sparsity': None,
            'ranks': channel_ranks
        })

    # structural search
    for name, module in module_list:
        logging.info(f'prune {name}: {tuple(module.weight.shape)}')
        prec_thre -= _prec_step

        # binary search for best sparsity
        min_t, max_t = 0.0, _max_sparsity

        # back up model params for the recovery after trials
        weight_backup = module.weight.detach().clone()
        if module.bias is None:
            bias_backup = None
        else:
            bias_backup = module.bias.detach().clone()

        mid_t = (min_t + max_t)/2
        logging.info('start sparisty search')
        for _ in range(_sparsity_search_step):
            mid_t = (min_t + max_t)/2
            module.sparsity = mid_t

            # restore the zero weight from make_pruning_effective
            module.weight[:] = weight_backup
            if bias_backup is None:
                module.bias = None
            else:
                module.bias[:] = bias_backup

            # restore channel mask
            module.prune_channel_mask[:] = False

            # update channel mask
            const_pruner.compute_mask(module)
            cap_pruner.compute_mask(module)

            # compensation
            try:
                update_weight(module)
            except LinAlgError:
                logging.info('stop searching for numerical error')
                min_t = 0
                break
            model = model.to(_device)

            # get model ready for precision test
            with make_pruning_effective(model) as pruend_model:
                # test for current precision
                prec = _test_env.test(
                    model=pruend_model, ratio=_test_ratio,
                    batch_size=_test_batch_size, device=_device
                )
                prec = prec['top1_prec']
            logging.info(
                f'sparsity: {mid_t:.4f}, prec: {prec:.4f}/{prec_thre:.4f}')

            # binary search
            if prec >= prec_thre:
                min_t = mid_t
            else:
                max_t = mid_t

        # after the trials, set the final sparsity
        module.sparsity = min_t
        logging.info(f'determined sparsity: {min_t:.4f}')

        # restore the zero weight from make_pruning_effective
        module.weight[:] = weight_backup
        if bias_backup is None:
            module.bias = None
        else:
            module.bias[:] = bias_backup

        # restore channel mask
        module.prune_channel_mask[:] = False

        # update channel mask
        const_pruner.compute_mask(module)
        cap_pruner.compute_mask(module)

        # compensation
        try:
            update_weight(module)
        except LinAlgError:
            logging.info('stop pruning this layer for numerical error')
            module.prune_channel_mask[:] = False
        finally:
            # there could be some appended bias terms on CPU
            model = model.to(_device)

            # mask associated weights
            module.to_prune = False

    # test for final precision
    with make_pruning_effective(model) as pruend_model:
        prec = _test_env.test(
            model=pruend_model, ratio=1.0,
            batch_size=_test_batch_size, device=_device
        )

    # test for final FLOPS drop
    infer_masks(model, dummy_input)
    pruned_flops = count_flops(model, dummy_input)
    flops_drop = (full_flops - pruned_flops) / full_flops
    for k in prec:
        k_prec = prec[k]
        logging.info(f'final precision {k}: {k_prec:.4f}')
    logging.info(f'final flops_drop: {flops_drop:.4f}')

    return model
