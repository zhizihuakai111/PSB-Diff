# PSB-Diff
## 项目简介
PSB-Diff 针对精准农业中大田监测对**秧苗行精确提取**的需求，缓解水面反光、覆膜、密度不均等复杂条件下全轮廓标注昂贵、像素级监督不稳定的问题，采用**根部点标注**作为弱监督信号：在影像上直接标定每株秧苗根系位置，避免冠层轮廓与真实根位之间的系统偏差。整体流程上，首先由**判别式语义分割器**在点监督下生成**粗糙先验掩码**，近似刻画秧苗根区的空间分布；继而引入**先验引导的 Bernoulli 扩散（PG-RLD）**，在二元掩码域内通过随时间递增的 Bernoulli 门控噪声注入，使中间状态在真值与先验之间混合演化，由粗先验约束正向扰动与反向重构的语义一致性。反向阶段采用**置信度引导的秧苗掩码去噪**：将当前掩码与原始影像送入去噪网络，利用像素预测偏离中性概率的程度构造置信度，并与时间步结合得到由先验向细化结果过渡的更新概率，末端以连续概率图经阈值得到二值掩码，从而在反光与低密度等区域抑制误传播、在高置信区域强化根位锐化。训练上联合**像素级二分类损失**、基于 Sobel 的**梯度域边界一致性**以及**质心保留损失**，在弱监督下兼顾边界形状与整体根区不发生漂移；推理上辅以**全局–局部协同机制**：全局阶段对去噪结果与置信图做动态阈值划分，对低置信区域裁切原分辨率局部块再次去噪并回融，以全局连贯性与局部精度兼顾复杂干扰。论文进一步在 UNet、DeepLab V3+、HRNet、SegFormer 等主流分割架构上作为先验生成器进行对比，验证扩散细化与不同骨干的兼容性。

在获得细化掩码之后，PSB-Diff 通过**拓扑约束的作物行提取**将离散秧苗中心连接为连续行线：由全部秧苗中心的最小面积外接矩形确定**全局主方向**，将点投影至主方向及其法向坐标系，在固定物理尺度网格内结合投影间距的密度与稳健统计设定**自适应行间阈值**，得到行片段并沿主方向做轻量合并以减少过度分割；对相邻网格采用**行级配准**（等行数时可用最优匹配，否则采用无冲突贪心策略），依据方向一致性、端点距离与对齐角等准则连接行段，得到大范围连续、平滑的**行中心线**。实验在黑龙江友谊、安徽塘池、广东江门等地无人机正射影像上开展，正射图由固定航高与 RTK 采集后经摄影测量生成；数据集将 orthomosaic 裁为 512×512 切片共约 3000 张，按区域均衡划分训练/验证/测试，标注为 COCO Keypoints 形式。定量与定性评测见正式论文及本文「部分结果展示」。

完整项目连接如下：通过网盘分享的文件：PSB-Diff.rar
链接: https://pan.baidu.com/s/1WbkCPq-6A41c_QIRVeXcYg?pwd=akki 提取码: akki 
--来自百度网盘超级会员v5的分享

## 部分结果展示
所提出的反向扩散过程在复杂田间场景中的秧苗掩膜渐进细化

<img width="886" height="448" alt="image" src="https://github.com/user-attachments/assets/5c2986a4-3639-48a3-af96-b98424d2f3cb" />

不同模型的水稻秧苗分割结果定性对比

<img width="886" height="707" alt="image" src="https://github.com/user-attachments/assets/87c026c3-4ab8-4924-b547-821a2e1f9325" />

四种复杂田间场景下的代表性原始图像及其真实标注

<img width="886" height="591" alt="image" src="https://github.com/user-attachments/assets/b92d4f98-6bcc-487e-9fa1-76229b4be079" />

不同模型计数误差和定位误差的箱线图与小提琴图对比

<img width="886" height="401" alt="image" src="https://github.com/user-attachments/assets/396efdc3-b049-40e5-b6a0-3ea6996e4af3" />

友谊、汤池和江门地区代表性的水稻秧苗种植行提取结果

<img width="886" height="472" alt="image" src="https://github.com/user-attachments/assets/6a9528ee-15e7-48c2-8541-a482fd2138bf" />

种植行提取偏差的累积概率分布

<img width="886" height="437" alt="image" src="https://github.com/user-attachments/assets/0c88d165-b845-4222-a319-e3eea240e805" />


## 环境与依赖
以下为复现本项目所用虚拟环境依赖

| 包名 | 版本 | 来源 |
|------|------|------|
| absl-py | 2.3.1 | PyPI |
| addict | 2.4.0 | PyPI |
| aliyun-python-sdk-core | 2.16.0 | PyPI |
| aliyun-python-sdk-kms | 2.16.5 | PyPI |
| bzip2 | 1.0.8 | conda-forge |
| ca-certificates | 2025.4.26 | conda-forge |
| cachetools | 5.5.2 | PyPI |
| certifi | 2025.4.26 | PyPI |
| cffi | 1.17.1 | PyPI |
| charset-normalizer | 3.4.2 | PyPI |
| click | 8.1.8 | PyPI |
| colorama | 0.4.6 | PyPI |
| contourpy | 1.1.1 | PyPI |
| crcmod | 1.7 | PyPI |
| cryptography | 45.0.3 | PyPI |
| cycler | 0.12.1 | PyPI |
| cython | 3.1.1 | PyPI |
| filelock | 3.14.0 | PyPI |
| fonttools | 4.57.0 | PyPI |
| future | 1.0.0 | PyPI |
| google-auth | 2.40.3 | PyPI |
| google-auth-oauthlib | 1.0.0 | PyPI |
| grpcio | 1.70.0 | PyPI |
| idna | 3.10 | PyPI |
| imageio | 2.35.1 | PyPI |
| importlib-metadata | 8.5.0 | PyPI |
| importlib-resources | 6.4.5 | PyPI |
| jmespath | 0.10.0 | PyPI |
| joblib | 1.4.2 | PyPI |
| kiwisolver | 1.4.7 | PyPI |
| lazy-loader | 0.4 | PyPI |
| libffi | 3.4.6 | conda-forge |
| liblzma | 5.8.1 | conda-forge |
| liblzma-devel | 5.8.1 | conda-forge |
| libsqlite | 3.50.0 | conda-forge |
| libzlib | 1.3.1 | conda-forge |
| markdown | 3.7 | PyPI |
| markdown-it-py | 3.0.0 | PyPI |
| markupsafe | 2.1.5 | PyPI |
| matplotlib | 3.7.5 | PyPI |
| mdurl | 0.1.2 | PyPI |
| mmcv | 2.2.0 | PyPI |
| mmcv-full | 1.7.1 | PyPI |
| mmengine | 0.10.7 | PyPI |
| model-index | 0.1.11 | PyPI |
| networkx | 3.1 | PyPI |
| numpy | 1.24.4 | PyPI |
| oauthlib | 3.3.1 | PyPI |
| opencv-python | 4.11.0.86 | PyPI |
| opendatalab | 0.0.10 | PyPI |
| openmim | 0.3.9 | PyPI |
| openssl | 3.5.0 | conda-forge |
| openxlab | 0.1.2 | PyPI |
| ordered-set | 4.1.0 | PyPI |
| oss2 | 2.17.0 | PyPI |
| packaging | 24.2 | PyPI |
| pandas | 2.0.3 | PyPI |
| pillow | 10.4.0 | PyPI |
| pip | 22.1.2 | conda-forge |
| platformdirs | 4.3.6 | PyPI |
| protobuf | 5.29.5 | PyPI |
| pyasn1 | 0.6.1 | PyPI |
| pyasn1-modules | 0.4.2 | PyPI |
| pycocotools | 2.0.7 | PyPI |
| pycparser | 2.22 | PyPI |
| pycryptodome | 3.23.0 | PyPI |
| pygments | 2.19.1 | PyPI |
| pyparsing | 3.1.4 | PyPI |
| python | 3.8.20 | conda-forge |
| python-dateutil | 2.9.0.post0 | PyPI |
| pytz | 2023.4 | PyPI |
| pywavelets | 1.4.1 | PyPI |
| pywin32 | 310 | PyPI |
| pyyaml | 6.0.2 | PyPI |
| regex | 2024.11.6 | PyPI |
| requests | 2.28.2 | PyPI |
| requests-oauthlib | 2.0.0 | PyPI |
| rich | 13.4.2 | PyPI |
| rsa | 4.9.1 | PyPI |
| scikit-image | 0.21.0 | PyPI |
| scikit-learn | 1.3.2 | PyPI |
| scipy | 1.10.1 | PyPI |
| seaborn | 0.13.2 | PyPI |
| setuptools | 60.2.0 | PyPI |
| six | 1.17.0 | PyPI |
| tabulate | 0.9.0 | PyPI |
| tensorboard | 2.14.0 | PyPI |
| tensorboard-data-server | 0.7.2 | PyPI |
| termcolor | 2.4.0 | PyPI |
| terminaltables | 3.1.10 | PyPI |
| threadpoolctl | 3.5.0 | PyPI |
| tifffile | 2023.7.10 | PyPI |
| tk | 8.6.13 | conda-forge |
| tomli | 2.2.1 | PyPI |
| torch | 1.12.1+cu113 | PyPI |
| torchaudio | 0.12.1+cu113 | PyPI |
| torchvision | 0.13.1+cu113 | PyPI |
| tqdm | 4.65.2 | PyPI |
| typing-extensions | 4.13.2 | PyPI |
| tzdata | 2025.2 | PyPI |
| ucrt | 10.0.22621.0 | conda-forge |
| urllib3 | 1.26.20 | PyPI |
| vc | 14.3 | conda-forge |
| vc14_runtime | 14.42.34438 | conda-forge |
| werkzeug | 3.0.6 | PyPI |
| wheel | 0.45.1 | conda-forge |
| xz | 5.8.1 | conda-forge |
| xz-tools | 5.8.1 | conda-forge |
| yapf | 0.40.1 | PyPI |
| zipp | 3.20.2 | PyPI |

### 训练与测试

1. **训练（示例：thin 配置 + 预生成粗糙掩码）**

```bash
python scripts/gen_coarse_masks_thin_only.py
python tools/train.py configs/segrefiner/segrefiner_thin_only.py
```
2. **测试**  
   - 先保证测试集对应目录下已有粗糙掩码（如 BIG 测试流程中先运行 `gen_coarse_masks_big.py`）。  
   - 调用 `tools/test.py`，指定配置、权重与输出目录，例如：

```bash
python tools/test.py configs/segrefiner/segrefiner_big_eval.py /path/to/latest.pth --out_dir big_test_results
```

   - 测试配置（如 `segrefiner_big_eval.py`）中的 `data_root`、`model_size`、`fine_prob_thr`、`centroid_correction` 等需与数据及实验设定一致。  
   - 语义测试路径下，`mmdet/apis/test.py` 中实现了水稻相关的 **F1/P/R、计数误差** 等指标汇总，结果可写入 CSV 等（视配置与调用方式而定）。

---

