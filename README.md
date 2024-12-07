## 安装依赖

```bash
pip install -r requirements.txt
```

## 下载

- 预训练模型下载

    > https://drive.google.com/file/d/1XGKcX2Zl_fIU9wPBDjTTxstOBfwQH8xc/view?usp=sharing

- 数据集下载

    > https://www.kaggle.com/datasets/rajendrabaskota/coco-fake    


## 使用

```bash
python train.py --cfg ./configs/COCO_2014.yaml
```

## 参考

@article{guo2022join,
  title={Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket},
  author={Guo, Nianhui and Bethge, Joseph and Meinel, Christoph and Yang, Haojin},
  journal={arXiv preprint arXiv:2211.12933},
  year={2022}
}
