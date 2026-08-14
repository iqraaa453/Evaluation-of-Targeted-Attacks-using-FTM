exp_configuration = {
    1: {
        'dataset': 'Medical-ChestXRay',  # Updated label
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iter': 300,
        'num_images': 10,
        'p': 1.,  # prob for DI

        # ✅ Updated Medical Target Models
        # Includes a mix of different datasets (NIH, CheXpert, PadChest) 
        # and different architectures (DenseNet, ResNet, ViT)
        'target_model_names': [
            'xrv_chexnet',        # DenseNet121 - All datasets
            'xrv_densenet_nih',   # DenseNet121 - NIH dataset
            'xrv_densenet_chex',  # DenseNet121 - CheXpert dataset
            'xrv_densenet_pc',    # DenseNet121 - PadChest dataset
            'xrv_resnet50',       # ResNet50 - Medical version
            'DenseNet121',        # Standard ImageNet (Cross-domain test)
            'ResNet50',           # Standard ImageNet (Cross-domain test)
            'vit_base_patch16_224', # Transformer (ImageNet)
            'efficientnet_b0'   # Edge model
        ],

        ####################################
        # FTM Parameters
        'ftm_beta': 0.01,
        'ftm_ensemble_size': 1,  # 1 for FTM, 2 for FTM-E
        'mix_prob': 0.1,
        'mix_upper_bound_feature': 0.75,

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # [mix_lower_bound_feature, mix_upper_bound_feature]
        'divisor': 4,
        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'Medical settings for RDI-TI-MI-FTM targeting XRV models'
    },
}