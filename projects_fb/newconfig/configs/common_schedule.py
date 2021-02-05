from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.solver import WarmupParamScheduler

from newconfig import LazyCall as L

lr_multiplier_1x = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[60000, 80000],
        num_updates=90000,
    ),
    warmup_length=1000 / 90000,
    warmup_method="linear",
    warmup_factor=0.001,
)

lr_multiplier_2x = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[120000, 160000],
        num_updates=180000,
    ),
    warmup_length=1000 / 180000,
    warmup_method="linear",
    warmup_factor=0.001,
)

lr_multiplier_3x = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[210000, 250000],
        num_updates=270000,
    ),
    warmup_length=1000 / 270000,
    warmup_method="linear",
    warmup_factor=0.001,
)

lr_multiplier_6x = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[480000, 520000],
        num_updates=540000,
    ),
    warmup_length=1000 / 540000,
    warmup_method="linear",
    warmup_factor=0.001,
)
