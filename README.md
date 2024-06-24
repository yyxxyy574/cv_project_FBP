# cv_project_FBP
## 基础任务
### 运行命令参数
`--train`：指定训练模式，在训练集上训练模型并保存在验证集上性能最好的模型

``--test``: 指定测试模式，在测试集上测试模型的性能并打印和保存相关指标

``--explain``: 指定解释模式，在给定的模型和图片上绘制热力图并保存

``--maml``: 指定个性化模式，使用个性化的数据集

``--person``: 指定个性化的用户，使用某个用户的个性化的数据集，默认为用户0

``--dataset``: 指定所用数据集，默认为fbp5500

``--save-name``: 指定结果保存时使用的文件名，结果包括性能最好的模型、测试指标、解释性热力图

`--load-from`: 指定加载模型的文件名，训练模式下可不指定，若指定了在该模型下继续训练，而测试和解释模型下必须指定

`--weight-classifier`: 分类分支的损失权重，默认为0.4

`--weight-regressor`: 回归分支的损失权重，默认为0.6

指定参数，运行main.py文件即可:
```bash
python main.py
```
### 实验脚本
探究分类分支权重对模型性能的影响实验:
```bash
./basic.sh
```
探究个性化数据集对模型性能的影响实验:
```bash
./personal.sh
```