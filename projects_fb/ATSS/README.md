# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

In this repository, we implement ATSS as part of RetinaNet

@article{zhang2019bridging,
  title   =  {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author  =  {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  journal =  {arXiv preprint arXiv:1912.02424},
  year    =  {2019}
}

## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/ATSS
python train_net.py --config-file configs/atss_R50_1x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/DeepLab
python train_net.py --config-file configs/atss_R50_1x.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## Results and Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">box AP</th>
<!-- TABLE BODY -->
 <tr><td align="left">ATSS</td>
<td align="center">R-50</td>
<td align="center">39.05</td>
</tr>
 <tr><td align="left">IOU top11</td>
<td align="center">R-50</td>
<td align="center">39.12</td>
</tr>
</tbody></table>
