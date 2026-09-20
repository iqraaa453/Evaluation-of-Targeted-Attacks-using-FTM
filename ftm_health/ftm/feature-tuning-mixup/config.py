exp_configuration = {
    1: {
        'dataset': 'Medical-ChestXRay',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iter': 300,
        'num_images': 1000,
        'p': 1.,  # prob for DI

        # Paper's Medical Target Models (Table III)
        # Surrogate: xrv_chexnet (DenseNet121 - all datasets)
        # Targets: 4 XRV models trained on different datasets
        'target_model_names': [
            'xrv_chexnet',        # Surrogate (white-box evaluation)
            'xrv_densenet_nih',   # DenseNet121 - NIH dataset
            'xrv_densenet_chex',  # DenseNet121 - CheXpert dataset
            'xrv_densenet_pc',    # DenseNet121 - PadChest dataset
            'xrv_resnet50',       # ResNet50 - Medical version (512x512)
        ],

        ####################################
        # FTM Parameters
        'ftm_beta': 0.01,
        'ftm_ensemble_size': 1,  # 1 for FTM, 2 for FTM-E
        'mix_prob': 0.1,
        'mix_upper_bound_feature': 0.75,

        'mixed_image_type_feature': 'C',
        'shuffle_image_feature': 'SelfShuffle',
        'blending_mode_feature': 'M',
        'mix_lower_bound_feature': 0.,
        'divisor': 4,
        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'Medical settings matching paper Table III: XRV ChexNet surrogate, 4 XRV targets'
    },
}