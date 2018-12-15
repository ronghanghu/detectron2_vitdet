Setup Cityscapes

### Steps to convert Cityscapes Annotations to COCO Format

1. Create symlinks for `cityscapes`:
```bash
cd /path/to/detectron2
mkdir -p datasets/cityscapes/annotations
mkdir -p datasets/cityscapes/images


for i in $(find /path/to/cityscapes/leftimg8bit/ -type f); \
    do ln -s $i /path/to/detectron2/datasets/cityscapes/images/ ; \
done

ln -s /path/to/cityscapes/gtFine /path/to/detectron2/datasets/cityscapes/
```

2. Use code from Detectron to perform the conversion

```bash
cd ~/github
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
python setup.py install

cd /path/to/detectron2
python tools/cityscapes/convert_cityscapes_to_coco.py \
       --datadir ~/datasets/cityscapes/ \
       --outdir datasets/cityscapes/annotations \
       --dataset cityscapes_instance_only
```
