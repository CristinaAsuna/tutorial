# 基础模型接口小课：一次下载，多篇教程复用

目标：分清“调用一个VLM回答问题”和“取出它的视觉/语言/逐层K/V供论文机制使用”。先用离线 tiny 理解接口，再用真实小模型验证；不需要为了每篇教程重写下载、预处理和加载代码。

共享实现位于 `general_utils/edu_core/edu_core/backbones/`。默认导入只需要 torch；`load_backbone("smolvlm")` 才加载 transformers。使用 `.capabilities` / `require_capabilities` 检查能力，不猜测所有模型都支持层间操作。

## 0. 环境与显式下载

在仓库根目录，用你的 PyTorch 环境执行：

```bash
python3 -m pip install -e 'general_utils/edu_core[pretrained,dev]'
PYTHONPATH=general_utils/edu_core python3 -m edu_core.backbones.download \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964
PYTHONPATH=general_utils/edu_core python3 mllm/backbones/run_pretrained_demo.py
```

模型是 [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)，Apache-2.0，含小型SigLIP视觉塔与SmolLM2语言部分。公开checkpoint的配置声明为 **Idefics3**，所以使用 `AutoModelForImageTextToText` 选择实际架构，而不是因产品名强行实例化另一种模型类。

下载仅发生在 download 命令，解析revision后保存精确SHA与缓存位置。普通加载全部 `local_files_only=True`，不会暗中联网或退回随机参数。权重放标准 Hugging Face cache，不写入Git；支持 `--cache-dir`，自定义缓存时后续用 `--model-path /完整/snapshots/SHA`。

本次已下载的snapshot为 `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964`。`tutorial-download.json` 位于该模型缓存目录，记录revision与本机路径。更换机器不需要修改课程代码。

默认CPU/float32/eager attention，不需要CUDA、FlashAttention或量化。当前测试版本为 torch 2.13.0、transformers 5.15.0、huggingface_hub 1.27.0。模型原始旧配置会触发一条嵌套 pad_token 警告；实际文本配置 pad=2、vocab=49280。适配器检查权重missing/unexpected/mismatched keys，任何不匹配直接失败；不屏蔽真正加载错误。

## 1. 输入如何变成视觉与文字表示

先做手算：两张图各4个视觉token，3个有效文字token，prefix共11个token。相机无效时，其4个token也必须全部无效，不能仅把图像置零。

在仓库根启动已设置 `PYTHONPATH=general_utils/edu_core` 的 Python：

```python
import torch
from edu_core.backbones import load_backbone, prefix_encoding

backbone = load_backbone("tiny")  # 不依赖网络或 transformers
images = torch.rand(1, 3, 8, 8)
raw = backbone.encode_vision(images)
projected = backbone.encode_vision(images, projected=True)
text = backbone.embed_text(torch.tensor([[1, 8, 9]]))
print(raw.tokens.shape, raw.grid, raw.cls_index)  # (1,4,24), (2,2), None
print(projected.tokens.shape, text.shape)         # (1,4,64), (1,3,64)
```

小练习：把两种tokens沿token轴拼接，预测其shape，再打印验证。不要把原始24维视觉特征与64维词嵌入直接拼在一起。

<details><summary>提示1：思路</summary>

只有投影到同一特征维度后，才能拼接序列轴。

</details>
<details><summary>提示2：操作</summary>

用 `projected.tokens`，调用 `torch.cat(...,dim=1)`；对应的valid mask也要拼接。

</details>
<details><summary>提示3：伪代码</summary>

`prefix = concat(projected_visual, text, axis=tokens)`；本例 `(1,7,64)`。

</details>

`VisionFeatures` 返回 `tokens/valid_mask/grid/cls_index`。默认是raw patch特征，`projected=True` 才进入语言空间。SmolVLM实际输出：raw `(1,1024,768)`、projected `(1,64,576)`，均没有CLS。

组件模式的图像输入是float RGB `[0,1]`，明确执行单张方形resize与SigLIP mean/std；不做原生processor的多裁剪扩张。**原生图文测试**使用官方processor/chat template，适合检查实际语言能力；**课程组件测试**保留该课程的projector/packing，适合检查学生机制，二者不能混同。

## 2. Mask、logits与生成

```python
prefix = torch.cat((projected.tokens, text), dim=1)
valid = torch.ones(prefix.shape[:2], dtype=torch.bool)
causal = backbone.language(prefix, valid)
allow = valid[:, :, None] & valid[:, None, :]
bidirectional = backbone.language(prefix, valid, allow_mask=allow)
print(causal.logits.shape)  # (B,L,V)
```

默认语言forward是causal，显式 `allow_mask=(B,L,L)` 才采用给定路由。True=允许，adapter内部将其转换为HF需要的4D additive mask。`position_ids` 默认按有效位cumsum，不随padding前进。`output_logits=False` 可避免π0无用的全词表输出。

小练习：固定模型，修改最后一个有效token的embedding。预测causal模式下第一个位置是否变化，再对比bidirectional模式。先猜结果再看检查器 `test_causal_vs_bidirectional_prefix_and_padding`。思路是“第一个query能否读取最后一个key”，不是输出shape。

完整接口：`language(embeddings,valid_mask,allow_mask=None,positions=None,use_cache=False,output_logits=True)` 返回 `LanguageOutput(hidden_states,logits,key_values)`。`generate(ids,max_new_tokens=4)` 是基础文本生成；真实图文问答使用SmolVLM的 `native_chat(image,question,max_new_tokens)`。

## 3. 获取逐层K/V并保留梯度

```python
cache = prefix_encoding(backbone, prefix, valid)
print(len(cache.key_values))
keys, values = cache.key_values[0]
print(keys.shape)  # tiny: (B,2,P,16)，SmolVLM: (B,3,P,64)
```

`PrefixEncoding` 包含每层K/V、valid mask和positions，key已经带RoPE。`DecoderSpec` 声明hidden size、层数、query/KV heads、head_dim、vocab和rope_theta；expert根据这些值构造注意力，不能假设每个模型都是32维。

小练习：令输入embedding需要梯度，对cache中key/value的平方均值反传。哪些参数应该有grad？分别测试tiny与frozen基础模型。参考操作是：输入 `.requires_grad_(True)` → `prefix_encoding` → loss → backward；冻结weights不会自动截断输入梯度。不要用 `no_grad` 包住需要训练projector的整个语言forward。

缓存只对同一次观测有效；读取时不追加suffix、不detach可训练路径。π0第4/7关进一步练习完整前向与缓存输出/梯度等价性。

## 4. 使用实际下载的模型

```python
from PIL import Image
backbone = load_backbone("smolvlm", frozen=True)  # 只读已下载缓存
image = Image.open("mllm/backbones/assets/red_square.png").convert("RGB")
print(backbone.native_chat(image, "What color is the square?", 12))
```

更换成自己的图片：

```bash
PYTHONPATH=general_utils/edu_core python3 mllm/backbones/run_pretrained_demo.py \
  --image /你的/图片.jpg --question 'Describe the image briefly.'
```

该脚本还检查raw/projected形状、30层cache、冻结参数和K/V对输入embedding的实际梯度。原生回复是模型输出，内容质量不能由shape测试保证；提供的红方块仅为简单本地fixture。

## 5. 三篇课程共用一份权重

```bash
PYTHONPATH=general_utils/edu_core python3 mllm/pi0/run_pi0_demo.py --backbone smolvlm
PYTHONPATH=general_utils/edu_core python3 mllm/llava/run_backbone_demo.py
PYTHONPATH=general_utils/edu_core python3 mllm/openvla/run_backbone_demo.py
```

均支持reference/practice选择；未完成practice报具体TODO。模型缓存跨课程共用，每个独立脚本只加载本进程需要的模型。冻结、loss、mask和生成策略仍由课程控制，不由loader偷偷决定。

- π0使用预训练的图像/文字及逐层K/V；CPU测试冻结基础模型、训练窄expert。
- LLaVA只复用视觉塔和语言模块，随机projector与学生packing仍执行。CPU桥接把raw patches平均池化为4×4，再补一个将被旧lesson剥离的合成mean/CLS槽；不声称SigLIP原生有CLS。vision/language wrappers注册不同子模块，冻结视觉不会连带冻结语言。
- OpenVLA只替换语言模块，两个视觉分支仍是toy。它的action vocabulary显式选256个非特殊ID，防止覆盖SmolVLM图像/控制token；通过codec包装学生量化函数，默认toy映射不变。

这些组件接线实验不表示LLaVA/OpenVLA已学会理解图像或控制机器人。所有默认toy检查保持离线；整仓库能力与后续边界见[接口审查](../../docs/backbone_audit.md)。
