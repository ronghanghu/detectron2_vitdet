

These are quick configs for performance or accuracy regression tracking purposes.

## Perf testing configs:

They are equivalent to the standard C4 / FPN models, only with extremely short schedules.

Metrics to look at:

```
INFO: Total training time: 0:3:20.276231
...
INFO: Total inference time: 0:01:20.276231
```


## Accuracy testing configs:

They are simplified versions of standard models, trained and tested on the same
minival dataset, with short schedules.

The schedule is designed to provide a stable enough mAP within minimal amount of training time.

Metrics to look at: mAPs.
