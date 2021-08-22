```
./detectron2/dev/fair/icebox ./detectron2/dev/fair/launch.py \
    --config-file ${config} \
    --num-machines 8 --num-gpus 8 -p learnlab --use-volta32 --name test
```