# Detectron2 Model Zoo and Baselines

## Introduction


### COCO Object Detection Baselines
<!--
To update the table in vim:
1. Remove the old table: d{
2. Copy the below command to the place of the table
3. :.!bash
./gen_html_table.py --config 'COCO-Detection/faster*50*'{1x,3x} 'COCO-Detection/faster*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_C4_1x -->
 <tr><td align="left">R50-C4</td>
<td align="center">1x</td>
<td align="center">0.593</td>
<td align="center">0.110</td>
<td align="center">4.8</td>
<td align="center">35.7</td>
<td align="center">137257644</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_1x -->
 <tr><td align="left">R50-DC5</td>
<td align="center">1x</td>
<td align="center">0.380</td>
<td align="center">0.089</td>
<td align="center">5.0</td>
<td align="center">37.3</td>
<td align="center">137847829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_1x -->
 <tr><td align="left">R50-FPN</td>
<td align="center">1x</td>
<td align="center">0.210</td>
<td align="center">0.060</td>
<td align="center">3.0</td>
<td align="center">37.9</td>
<td align="center">137257794</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_C4_3x -->
 <tr><td align="left">R50-C4</td>
<td align="center">3x</td>
<td align="center">0.589</td>
<td align="center">0.108</td>
<td align="center">4.8</td>
<td align="center">38.4</td>
<td align="center">137849393</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_3x -->
 <tr><td align="left">R50-DC5</td>
<td align="center">3x</td>
<td align="center">0.378</td>
<td align="center">0.095</td>
<td align="center">5.0</td>
<td align="center">39.0</td>
<td align="center">137849425</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_3x -->
 <tr><td align="left">R50-FPN</td>
<td align="center">3x</td>
<td align="center">0.209</td>
<td align="center">0.058</td>
<td align="center">3.0</td>
<td align="center">40.2</td>
<td align="center">137849458</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_C4_3x -->
 <tr><td align="left">R101-C4</td>
<td align="center">3x</td>
<td align="center">0.656</td>
<td align="center">0.137</td>
<td align="center">5.9</td>
<td align="center">41.1</td>
<td align="center">138204752</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_DC5_3x -->
 <tr><td align="left">R101-DC5</td>
<td align="center">3x</td>
<td align="center">0.452</td>
<td align="center">0.103</td>
<td align="center">6.1</td>
<td align="center">40.6</td>
<td align="center">138204841</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_FPN_3x -->
 <tr><td align="left">R101-FPN</td>
<td align="center">3x</td>
<td align="center">0.286</td>
<td align="center">0.071</td>
<td align="center">4.1</td>
<td align="center">42.0</td>
<td align="center">137851257</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left">X101-FPN</td>
<td align="center">3x</td>
<td align="center">0.638</td>
<td align="center">0.139</td>
<td align="center">6.7</td>
<td align="center">43.0</td>
<td align="center">139173657</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/metrics.json">metrics</a></td>
</tr>
</tbody></table>



### COCO Instance Segmentation Baselines
<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask*50*'{1x,3x} 'COCO-InstanceSegmentation/mask*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_C4_1x -->
 <tr><td align="left">R50-C4</td>
<td align="center">1x</td>
<td align="center">0.621</td>
<td align="center">0.140</td>
<td align="center">5.2</td>
<td align="center">36.8</td>
<td align="center">32.2</td>
<td align="center">137259246</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_1x -->
 <tr><td align="left">R50-DC5</td>
<td align="center">1x</td>
<td align="center">0.471</td>
<td align="center">0.126</td>
<td align="center">6.5</td>
<td align="center">38.3</td>
<td align="center">34.2</td>
<td align="center">137260150</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left">R50-FPN</td>
<td align="center">1x</td>
<td align="center">0.261</td>
<td align="center">0.087</td>
<td align="center">3.4</td>
<td align="center">38.6</td>
<td align="center">35.2</td>
<td align="center">137260431</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_C4_3x -->
 <tr><td align="left">R50-C4</td>
<td align="center">3x</td>
<td align="center">0.622</td>
<td align="center">0.137</td>
<td align="center">5.2</td>
<td align="center">39.8</td>
<td align="center">34.4</td>
<td align="center">137849525</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_3x -->
 <tr><td align="left">R50-DC5</td>
<td align="center">3x</td>
<td align="center">0.470</td>
<td align="center">0.111</td>
<td align="center">6.5</td>
<td align="center">40.0</td>
<td align="center">35.9</td>
<td align="center">137849551</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left">R50-FPN</td>
<td align="center">3x</td>
<td align="center">0.261</td>
<td align="center">0.079</td>
<td align="center">3.4</td>
<td align="center">41.0</td>
<td align="center">37.2</td>
<td align="center">137849600</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_C4_3x -->
 <tr><td align="left">R101-C4</td>
<td align="center">3x</td>
<td align="center">0.691</td>
<td align="center">0.163</td>
<td align="center">6.3</td>
<td align="center">42.6</td>
<td align="center">36.7</td>
<td align="center">138363239</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_DC5_3x -->
 <tr><td align="left">R101-DC5</td>
<td align="center">3x</td>
<td align="center">0.545</td>
<td align="center">0.129</td>
<td align="center">7.6</td>
<td align="center">41.9</td>
<td align="center">37.3</td>
<td align="center">138363294</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_FPN_3x -->
 <tr><td align="left">R101-FPN</td>
<td align="center">3x</td>
<td align="center">0.340</td>
<td align="center">0.092</td>
<td align="center">4.6</td>
<td align="center">42.9</td>
<td align="center">38.6</td>
<td align="center">138205316</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a></td>
</tr>
</tbody></table>

