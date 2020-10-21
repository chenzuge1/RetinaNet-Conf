# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # step=[17, 20])
    # step=[14, 17])
    # step=[11, 14])
    # step=[8, 11])
    step=[16, 22])
total_epochs = 24
# total_epochs = 12
# total_epochs = 15
# total_epochs = 18
# total_epochs = 21
