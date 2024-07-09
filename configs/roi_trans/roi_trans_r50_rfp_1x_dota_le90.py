_base_ = ['./roi_trans_r50_fpn_1x_dota_le90.py']

model = dict(
    type='FRAN_RoITransformer',
    backbone=dict(
        type='DetectoRS_ResNet',
        rfp_inplanes=256,
        conv_cfg=dict(type='ConvAWS'),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        rfp_sharing=True,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1)))