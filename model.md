
# 模块列表

#### 模块代码统一在：ultralytics/nn/extra_modules文件夹下，对应的论文都在每个模块的py文件头部标识清楚
#### 配置文件都在：ultralytics/cfg/models/improve，部分模块py没有对应的yaml，这些请在GuideVideo-MG.md里查阅教程，都是只需改yaml

# attention
- ultralytics/nn/extra_modules/attention(配置文件在ultralytics/cfg/models/improve/attention)

    1. ultralytics/nn/extra_modules/attention/SEAM.py
    2. CVPR2021|ultralytics/nn/extra_modules/attention/ca.py
    3. ICASSP2023|ultralytics/nn/extra_modules/attention/ema.py
    4. ICML2021|ultralytics/nn/extra_modules/attention/simam.py
    5. ICCV2023|ultralytics/nn/extra_modules/attention/lsk.py
    6. WACV2024|ultralytics/nn/extra_modules/attention/DeformableLKA.py
    7. ultralytics/nn/extra_modules/attention/mlca.py
    8. BIBM2024|ultralytics/nn/extra_modules/attention/FSA.py
    9. AAAI2025|ultralytics/nn/extra_modules/attention/CDFA.py
    10. TGRS2025|ultralytics/nn/extra_modules/attention/MCA.py
    11. CVPR2025|ultralytics/nn/extra_modules/attention/CASAB.py
    12. NN2025|ultralytics/nn/extra_modules/attention/KSFA.py
    13. TGRS2025|ultralytics/nn/extra_modules/attention/ACA.py
    14. TGRS2025|ultralytics/nn/extra_modules/attention/DHPF.py
    15. TGRS2025|ultralytics/nn/extra_modules/attention/ACAB.py
    16. 自研模块|ultralytics/nn/extra_modules/attention/FSDA.py

# conv_module
- ultralytics/nn/extra_modules/conv_module(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第五节,支持与attention部分联合改进CSP模块中的残差块)

    1. CVPR2021|ultralytics/nn/extra_modules/conv_module/dbb.py
    2. TIP2024|ultralytics/nn/extra_modules/conv_module/deconv.py
    3. ICCV2023|ultralytics/nn/extra_modules/conv_module/dynamic_snake_conv.py
    4. CVPR2023|ultralytics/nn/extra_modules/conv_module/pconv.py
    5. AAAI2025|ultralytics/nn/extra_modules/conv_module/psconv.py
    6. CVPR2025|ultralytics/nn/extra_modules/conv_module/ShiftwiseConv.py
    7. ultralytics/nn/extra_modules/conv_module/wdbb.py
    8. ultralytics/nn/extra_modules/conv_module/deepdbb.py
    9. ECCV2024|ultralytics/nn/extra_modules/conv_module/wtconv2d.py
    10. CVPR2023|ultralytics/nn/extra_modules/conv_module/ScConv.py
    11. ultralytics/nn/extra_modules/conv_module/dcnv2.py
    12. CVPR2024|ultralytics/nn/extra_modules/conv_module/DilatedReparamConv.py
    13. ultralytics/nn/extra_modules/conv_module/gConv.py
    14. CVPR2024|ultralytics/nn/extra_modules/conv_module/IDWC.py
    15. ultralytics/nn/extra_modules/conv_module/DSA.py
    16. CVPR2025|ultralytics/nn/extra_modules/conv_module/FDConv.py
    17. CVPR2023|ultralytics/nn/extra_modules/conv_module/dcnv3.py
    18. CVPR2024|ultralytics/nn/extra_modules/conv_module/dcnv4.py
    19. CVPR2024|ultralytics/nn/extra_modules/conv_module/DynamicConv.py
    20. CVPR2024|ultralytics/nn/extra_modules/conv_module/FADC.py
    21. CVPR2023|ultralytics/nn/extra_modules/conv_module/SMPConv.py
    22. MIA2025|ultralytics/nn/extra_modules/conv_module/FourierConv.py
    23. CVPR2024|ultralytics/nn/extra_modules/conv_module/SFSConv.py
    24. ICCV2025|ultralytics/nn/extra_modules/conv_module/ConvAttn.py
    25. ICCV2025|ultralytics/nn/extra_modules/conv_module/Converse2D.py
    26. CVPR2025|ultralytics/nn/extra_modules/conv_module/gcconv.py
    27. ACCV2024|ultralytics/nn/extra_modules/conv_module/RMBC.py
    28. CVPR2026|ultralytics/nn/extra_modules/conv_module/DEGConv.py
    29. TGRS2026|ultralytics/nn/extra_modules/conv_module/CKConv.py
    30. 自研模块|ultralytics/nn/extra_modules/conv_module/AGIDWC.py
    31. 自研模块|ultralytics/nn/extra_modules/conv_module/MSIDWC.py
    32. ICML2026|ultralytics/nn/extra_modules/conv_module/PolyConv.py
    33. 自研模块|ultralytics/nn/extra_modules/conv_module/FAKConv.py
    34. TGRS2026|ultralytics/nn/extra_modules/conv_module/EDDConv.py

- engine/extre_module/custom_nn/stem(配置文件在ultralytics/cfg/models/improve/stem)

    1. ultralytics/nn/extra_modules/stem/SRFD.py
    2. ultralytics/nn/extra_modules/stem/LoG.py
    3. ICCV2023|ultralytics/nn/extra_modules/stem/RepStem.py
    4. 自研模块|ultralytics/nn/extra_modules/stem/ODALStem.py
    5. 自研模块|ultralytics/nn/extra_modules/stem/DPAS.py
    6. 自研模块|ultralytics/nn/extra_modules/stem/WGFS.py

# upsample
- ultralytics/nn/extra_modules/upsample(配置文件在ultralytics/cfg/models/improve/upsample)

    1. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb.py
    2. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb_sc.py
    3. ultralytics/nn/extra_modules/upsample/WaveletUnPool.py
    4. ICCV2019|ultralytics/nn/extra_modules/upsample/CARAFE.py
    5. ICCV2023|ultralytics/nn/extra_modules/upsample/DySample.py
    6. ICCV2025|ultralytics/nn/extra_modules/upsample/Converse2D_Up.py
    7. CVPR2025|ultralytics/nn/extra_modules/upsample/DSUB.py
    8. 自研模块|ultralytics/nn/extra_modules/upsample/FEUCB.py
    9. 自研模块|ultralytics/nn/extra_modules/upsample/DOGEUCB.py

# downsample
- ultralytics/nn/extra_modules/downsample(配置文件在ultralytics/cfg/models/improve/downsample)

    1. TIP2020|ultralytics/nn/extra_modules/downsample/gcnet.py
    2. 自研模块|ultralytics/nn/extra_modules/downsample/lawds.py
    3. ultralytics/nn/extra_modules/downsample/WaveletPool.py
    4. ultralytics/nn/extra_modules/downsample/ADown.py
    5. ultralytics/nn/extra_modules/downsample/YOLOV7Down.py
    6. ultralytics/nn/extra_modules/downsample/SPDConv.py
    7. ultralytics/nn/extra_modules/downsample/HWD.py
    8. ultralytics/nn/extra_modules/downsample/DRFD.py
    9. TGRS2025|ultralytics/nn/extra_modules/conv_module/FSConv.py
    10. 自研模块|ultralytics/nn/extra_modules/downsample/EdgeLAWDS.py
    11. 自研模块|ultralytics/nn/extra_modules/downsample/FreqLAWDS.py
    12. 自研模块|ultralytics/nn/extra_modules/downsample/RouterLAWDS.py
    13. 自研模块|ultralytics/nn/extra_modules/downsample/FSCGD.py
    14. TGRS2026|ultralytics/nn/extra_modules/downsample/DMSSP.py
    15. TGRS2026|ultralytics/nn/extra_modules/downsample/MSAD.py
    16. CVPR2026|ultralytics/nn/extra_modules/downsample/WaveletDownsample.py

# csp-module / module
- ultralytics/nn/extra_modules/module(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. AAAI2025|ultralytics/nn/extra_modules/module/APBottleneck.py
    2. CVPR2025|ultralytics/nn/extra_modules/module/efficientVIM.py
    3. CVPR2023|ultralytics/nn/extra_modules/module/fasterblock.py
    4. CVPR2024|ultralytics/nn/extra_modules/module/starblock.py
    5. ultralytics/nn/extra_modules/module/DWR.py
    6. CVPR2024|ultralytics/nn/extra_modules/module/UniRepLKBlock.py
    7. CVPR2025|ultralytics/nn/extra_modules/module/mambaout.py
    8. AAAI2024|ultralytics/nn/extra_modules/module/DynamicFilter.py
    9. ultralytics/nn/extra_modules/module/StripBlock.py
    10. TGRS2024|ultralytics/nn/extra_modules/module/elgca.py
    11. CVPR2024|ultralytics/nn/extra_modules/module/LEGM.py
    12. ICCV2023|ultralytics/nn/extra_modules/module/iRMB.py
    13. TPAMI2025|ultralytics/nn/extra_modules/module/MSBlock.py
    14. ICLR2024|ultralytics/nn/extra_modules/module/FATBlock.py
    15. CVPR2024|ultralytics/nn/extra_modules/module/MSCB.py
    16. ultralytics/nn/extra_modules/module/LEGBlock.py
    17. ultralytics/nn/extra_modules/module/GLSA.py
    18. CVPR2025|ultralytics/nn/extra_modules/module/RCB.py
    19. ECCV2024|ultralytics/nn/extra_modules/module/JDPM.py
    20. CVPR2025|ultralytics/nn/extra_modules/module/vHeat.py
    21. CVPR2025|ultralytics/nn/extra_modules/module/EBlock.py
    22. CVPR2025|ultralytics/nn/extra_modules/module/DBlock.py
    23. ECCV2024|ultralytics/nn/extra_modules/module/FMB.py
    24. CVPR2024|ultralytics/nn/extra_modules/module/IDWB.py
    25. ECCV2022|ultralytics/nn/extra_modules/module/LFE.py
    26. AAAI2025|ultralytics/nn/extra_modules/module/FCM.py
    27. CVPR2024|ultralytics/nn/extra_modules/module/RepViTBlock.py
    28. CVPR2024|ultralytics/nn/extra_modules/module/PKIModule.py
    29. CVPR2024|ultralytics/nn/extra_modules/module/camixer.py
    30. ICCV2025|ultralytics/nn/extra_modules/module/ESC.py
    31. TGRS2025|ultralytics/nn/extra_modules/module/ARF.py
    32. AAAI2024|ultralytics/nn/extra_modules/module/CFBlock.py
    33. IJCV2024|ultralytics/nn/extra_modules/module/FMA.py
    34. ultralytics/nn/extra_modules/module/LWGA.py
    35. TGRS2025|ultralytics/nn/extra_modules/module/CSSC.py
    36. TGRS2025|ultralytics/nn/extra_modules/module/CNCM.py
    37. ICCV2025|ultralytics/nn/extra_modules/module/HFRB.py
    38. ICIP2025|ultralytics/nn/extra_modules/module/EVA.py
    39. CVPR2025|ultralytics/nn/extra_modules/module/IEL.py
    40. MICCAI2023|ultralytics/nn/extra_modules/module/MFEBlock.py
    41. AAAI2026|ultralytics/nn/extra_modules/module/PartialNetBlock.py
    42. TGRS2025|ultralytics/nn/extra_modules/module/DRG.py
    43. ultralytics/nn/extra_modules/module/Wave2D.py
    44. TGRS2025|ultralytics/nn/extra_modules/module/GLGM.py
    45. TGRS2025|ultralytics/nn/extra_modules/module/MAC.py
    46. AAAI2026|ultralytics/nn/extra_modules/module/SPJFB.py
    47. 自研模块|ultralytics/nn/extra_modules/module/FasterCGABlock.py
    48. CVPR2026|ultralytics/nn/extra_modules/module/sparse_mamba_block.py
    49. CVPR2026|ultralytics/nn/extra_modules/module/MSInit.py
    50. CVPR2026|ultralytics/nn/extra_modules/module/PFG.py
    51. CVPR2026|ultralytics/nn/extra_modules/module/LFP.py
    52. 自研模块|ultralytics/nn/extra_modules/module/AMSI.py
    53. 自研模块|ultralytics/nn/extra_modules/module/CMSI.py
    54. 自研模块|ultralytics/nn/extra_modules/module/FMSI.py
    55. 自研模块|ultralytics/nn/extra_modules/module/HPFGA.py
    56. 自研模块|ultralytics/nn/extra_modules/module/DPFGA.py
    57. 自研模块|ultralytics/nn/extra_modules/module/HOIE.py
    58. 自研模块|ultralytics/nn/extra_modules/module/ADIE.py
    59. TGRS2025|ultralytics/nn/extra_modules/module/DSEBlock.py
    60. TGRS2025|ultralytics/nn/extra_modules/module/LaSEA.py
    61. CVPR2026|ultralytics/nn/extra_modules/module/SFEB.py
    62. TIP2026|ultralytics/nn/extra_modules/module/FourierSR.py
    63. CVPR2026|ultralytics/nn/extra_modules/module/FrequencyCM.py
    64. AAAI2026|ultralytics/nn/extra_modules/module/RFGM.py
    65. CVPR2026|ultralytics/nn/extra_modules/module/IRA.py
    66. ECCV2026|ultralytics/nn/extra_modules/module/PKIBlockV2.py
    67. 自研模块|ultralytics/nn/extra_modules/module/BCIM.py
    68. 自研模块|ultralytics/nn/extra_modules/module/DGCM.py
    69. 自研模块|ultralytics/nn/extra_modules/module/EFCM.py
    70. ICML2026|ultralytics/nn/extra_modules/module/AMCM.py

# block
- ultralytics/nn/extra_modules/block (此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. ultralytics/nn/extra_modules/block/CSPBlock.py
    2. TPAMI2025|ultralytics/nn/extra_modules/block/MANet.py
    3. TPAMI2024|ultralytics/nn/extra_modules/block/MetaFormer.py
    4. ultralytics/nn/extra_modules/block/RepHMS.py
    5. 自研模块|ultralytics/nn/extra_modules/block/CBIMS.py

# transformer
- ultralytics/nn/extra_modules/transformer(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. ICLR2025|ultralytics/nn/extra_modules/transformer/PolaLinearAttention.py
    2. CVPR2023|ultralytics/nn/extra_modules/transformer/biformer.py
    3. CVPR2023|ultralytics/nn/extra_modules/transformer/CascadedGroupAttention.py
    4. CVPR2022|ultralytics/nn/extra_modules/transformer/DAttention.py
    5. ICLR2022|ultralytics/nn/extra_modules/transformer/DPBAttention.py
    6. CVPR2024|ultralytics/nn/extra_modules/transformer/AdaptiveSparseSA.py
    7. ultralytics/nn/extra_modules/transformer/GSA.py
    8. ultralytics/nn/extra_modules/transformer/RSA.py
    9. ECCV2024|ultralytics/nn/extra_modules/transformer/FSSA.py
    10. AAAI2025|ultralytics/nn/extra_modules/transformer/DilatedGCSA.py
    11. AAAI2025|ultralytics/nn/extra_modules/transformer/DilatedMWSA.py
    12. CVPR2024|ultralytics/nn/extra_modules/transformer/SHSA.py
    13. IJCAI2024|ultralytics/nn/extra_modules/transformer/CTA.py
    14. IJCAI2024|ultralytics/nn/extra_modules/transformer/SFA.py
    15. ultralytics/nn/extra_modules/transformer/MSLA.py
    16. ACMMM2025|ultralytics/nn/extra_modules/transformer/CPIA_SA.py
    17. NN2025|ultralytics/nn/extra_modules/transformer/TokenSelectAttention.py
    18. CVPR2025|ultralytics/nn/extra_modules/transformer/TAB.py
    19. TPAMI2025|ultralytics/nn/extra_modules/transformer/LRSA.py
    20. ICCV2025|ultralytics/nn/extra_modules/transformer/MALA.py
    21. ICML2023|ultralytics/nn/extra_modules/transformer/MUA.py
    22. ACMMM2025|ultralytics/nn/extra_modules/transformer/EGSA.py
    23. ACMMM2025|ultralytics/nn/extra_modules/transformer/SWSA.py
    24. AAAI2026|ultralytics/nn/extra_modules/transformer/DHOGSA.py
    25. NeurIPS2025|ultralytics/nn/extra_modules/transformer/CBSA.py
    26. TGRS2025|ultralytics/nn/extra_modules/transformer/DPWA.py
    27. TIP2025|ultralytics/nn/extra_modules/transformer/DWM_MSA.py
    28. CVPR2026|ultralytics/nn/extra_modules/transformer/BinaryAttention.py
    29. CVPR2025|ultralytics/nn/extra_modules/transformer/wca.py
    30. TGRS2026|ultralytics/nn/extra_modules/transformer/CGTA.py
    31. TGRS2026|ultralytics/nn/extra_modules/transformer/LCGA.py
    32. AAAI2026|ultralytics/nn/extra_modules/transformer/CirculantAttention.py
    33. CVPR2026|ultralytics/nn/extra_modules/transformer/WDAM.py
    34. TGRS2026|ultralytics/nn/extra_modules/transformer/GLCDM.py
    35. ICIP2026|ultralytics/nn/extra_modules/transformer/SDTA.py
    36. CVPR2026F|ultralytics/nn/extra_modules/transformer/SAA.py
    37. TGRS2026|ultralytics/nn/extra_modules/transformer/MSAM.py
    38. TCSVT2026|ultralytics/nn/extra_modules/transformer/ACE.py
    39. CVPR2026|ultralytics/nn/extra_modules/transformer/SpatialEnhancedAttention.py
    40. CVPR2026|ultralytics/nn/extra_modules/transformer/WACGA.py

# mamba
- ultralytics/nn/extra_modules/mamba(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. AAAI2025|ultralytics/nn/extra_modules/mamba/SS2D.py
    2. CVPR2025|ultralytics/nn/extra_modules/mamba/ASSM.py
    3. CVPR2025|ultralytics/nn/extra_modules/mamba/SAVSS.py
    4. CVPR2025|ultralytics/nn/extra_modules/mamba/MobileMamba/mobilemamba.py
    5. CVPR2025|ultralytics/nn/extra_modules/mamba/MaIR.py
    6. TGRS2025|ultralytics/nn/extra_modules/mamba/GLVSS.py
    7. ICCV2025|ultralytics/nn/extra_modules/mamba/VSSD.py
    8. ICCV2025|ultralytics/nn/extra_modules/mamba/TinyViM.py
    9. INFFUS2025|ultralytics/nn/extra_modules/mamba/CSI.py
    10. TIP2025|ultralytics/nn/extra_modules/mamba/SFMB.py
    11. TGRS2025|ultralytics/nn/extra_modules/mamba/GLSS.py
    12. TGRS2025|ultralytics/nn/extra_modules/mamba/GLSS2D.py
    13. CVPR2026|ultralytics/nn/extra_modules/mamba/TransMixer.py
    14. CVPR2026|ultralytics/nn/extra_modules/mamba/sparse_state_space.py
    15. AAAI2026|ultralytics/nn/extra_modules/mamba/GradMamba.py
    16. AAAI2026|ultralytics/nn/extra_modules/mamba/PatchMamba.py

# mlp
- ultralytics/nn/extra_modules/mlp(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. CVPR2024|ultralytics/nn/extra_modules/mlp/ConvolutionalGLU.py
    2. IJCAI2024|ultralytics/nn/extra_modules/mlp/DFFN.py
    3. ICLR2024|ultralytics/nn/extra_modules/mlp/FMFFN.py
    4. CVPR2024|ultralytics/nn/extra_modules/mlp/FRFN.py
    5. ECCV2024|ultralytics/nn/extra_modules/mlp/EFFN.py
    6. WACV2025|ultralytics/nn/extra_modules/mlp/SEFN.py
    7. ICLR2025|ultralytics/nn/extra_modules/mlp/KAN.py
    8. CVPR2025|ultralytics/nn/extra_modules/mlp/EDFFN.py
    9. ICVJ2024|ultralytics/nn/extra_modules/mlp/DML.py
    10. AAAI2026|ultralytics/nn/extra_modules/mlp/DIFF.py
    11. 自研模块|ultralytics/nn/extra_modules/mlp/MCCG.py
    12. 自研模块|ultralytics/nn/extra_modules/mlp/DSRG.py
    13. CVPR2026|ultralytics/nn/extra_modules/mlp/AFFN.py
    14. 自研模块|ultralytics/nn/extra_modules/mlp/MSAFFN.py
    15. 自研模块|ultralytics/nn/extra_modules/mlp/GLSFFN.py
    16. ICML2026|ultralytics/nn/extra_modules/mlp/PolyMLP.py

# neck
- ultralytics/nn/extra_modules/neck(配置文件在ultralytics/cfg/models/improve/neck)

    1. ultralytics/nn/extra_modules/neck/ASF.py
    2. ultralytics/nn/extra_modules/neck/BiFPN.py
    3. AAAI2022|ultralytics/nn/extra_modules/neck/CTrans.py
    4. ultralytics/nn/extra_modules/neck/EfficientRepBiPAN.py
    5. ultralytics/nn/extra_modules/neck/GFPN.py
    6. ultralytics/nn/extra_modules/neck/HSFPN.py
    7. AAAI2025|ultralytics/nn/extra_modules/neck/HS_FPN.py
    8. TPAMI2025|ultralytics/nn/extra_modules/neck/HyperComputeModule.py
    9. ultralytics/nn/extra_modules/neck/SlimNeck.py
    10. ultralytics/nn/extra_modules/neck/GoldYOLO.py
    11. ultralytics/nn/extra_modules/neck/EMBSFPN.py
    12. ultralytics/nn/extra_modules/neck/FDPN.py((里面有三个自研模块FocusFeature、DynamicFrequencyFocusFeature、AlignmentGuidedFocusFeature))
    13. PR2026|ultralytics/nn/extra_modules/neck/A3FPN.py

# featurefusion
- ultralytics/nn/extra_modules/featurefusion(配置文件在ultralytics/cfg/models/improve/featurefusion)

    1. 自研模块|ultralytics/nn/extra_modules/featurefusion/cgfm.py
    2. BMVC2024|ultralytics/nn/extra_modules/featurefusion/msga.py
    3. CVPR2024|ultralytics/nn/extra_modules/featurefusion/mfm.py
    4. TIP2023|ultralytics/nn/extra_modules/featurefusion/CSFCN.py
    5. BIBM2024|ultralytics/nn/extra_modules/featurefusion/mpca.py
    6. ACMMM2024|ultralytics/nn/extra_modules/featurefusion/wfu.py
    7. CVPR2025|ultralytics/nn/extra_modules/featurefusion/GDSAFusion.py
    8. ultralytics/nn/extra_modules/featurefusion/PST.py
    9. TGRS2025|ultralytics/nn/extra_modules/featurefusion/MSAM.py
    10. INFFUS2025|ultralytics/nn/extra_modules/featurefusion/DPCF.py
    11. CVRP2025|ultralytics/nn/extra_modules/featurefusion/LCA.py
    12. TGRS2025|ultralytics/nn/extra_modules/featurefusion/HFFE.py
    13. TGRS2025|ultralytics/nn/extra_modules/featurefusion/MFPM.py
    14. TGRS2025|ultralytics/nn/extra_modules/featurefusion/ERM.py
    15. TIP2025|ultralytics/nn/extra_modules/featurefusion/CAFM.py
    16. TIP2024|ultralytics/nn/extra_modules/featurefusion/CGAFusion.py
    17. IF2023|ultralytics/nn/extra_modules/featurefusion/PSFM.py
    18. IF2023|ultralytics/nn/extra_modules/featurefusion/SDFM.py
    19. 自研模块|ultralytics/nn/extra_modules/featurefusion/DAF.py
    20. 自研模块|ultralytics/nn/extra_modules/featurefusion/CIDAF.py
    21. 自研模块|ultralytics/nn/extra_modules/featurefusion/WDAF.py
    22. CVPR2026|ultralytics/nn/extra_modules/featurefusion/SFSFusion.py
    23. CVPR2026|ultralytics/nn/extra_modules/featurefusion/FAAFusion.py
    24. PR2026|ultralytics/nn/extra_modules/featurefusion/HAFFormer.py
    25. 自研模块|ultralytics/nn/extra_modules/featurefusion/LowFrequencyFeatureFusion.py
    26. 自研模块|ultralytics/nn/extra_modules/featurefusion/DCGRM.py
    27. 自研模块|ultralytics/nn/extra_modules/featurefusion/MSCRM.py
    28. TPAMI2025|ultralytics/nn/extra_modules/featurefusion/LPRM.py
    29. TCSVT2026|ultralytics/nn/extra_modules/featurefusion/MAFusion.py
    30. TGRS2026|ultralytics/nn/extra_modules/featurefusion/CFDM.py
    31. TGRS2026|ultralytics/nn/extra_modules/featurefusion/WFAM.py

# norm
- ultralytics/nn/extra_modules/norm(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. ICML2024|engine/extre_module/custom_nn/transformer/repbn.py
    2. CVPR2025|engine/extre_module/custom_nn/transformer/dyt.py
    3. engine/extre_module/custom_nn/norm/derf.py

# featurepreprocess
- ultralytics/nn/extra_modules/featurepreprocess(配置文件在ultralytics/cfg/models/improve/featurepreprocess)

    1. TGRS2025|ultralytics/nn/extra_modules/featurepreprocess/FAENet.py

# head
- ultralytics/nn/extra_modules/head(配置文件在ultralytics/cfg/models/improve/head)

    1. ultralytics/nn/extra_modules/head/LSPCD.py
    2. ultralytics/nn/extra_modules/head/LQE.py
    3. ultralytics/nn/extra_modules/head/DyHead.py(需要安装好mmcv)