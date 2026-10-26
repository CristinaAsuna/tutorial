# 基础模型依赖与复用审查（2026-10-16）

范围：检查 mllm、generative_model、general_utils 下的Python入口、notebook中的from_pretrained/AutoModel/timm/torch.hub调用，并对照各课程构造器和机制。现有课程以自写toy为主，不能把所有小Transformer都替换成一个VLM `generate()`。

本次共享入口为 `edu_core.backbones`，支持视觉tokens、文字embedding/logits、显式mask和逐层prefix K/V；默认tiny仅依赖torch，预训练依赖可选。π0、LLaVA、OpenVLA已有可运行的本地SmolVLM接入，其余保持原填答和接口。

|课程/领域|实际需要的能力|本次处理与边界|
|---|---|---|
|π0|多图视觉、文字嵌入、分层K/V、RoPE/GQA几何、块状mask|完整tiny机制；真实SmolVLM前缀+30层专家接线；冻结模式和端到端tiny区分|
|LLaVA|raw视觉patch、可接受inputs_embeds的LM、logits|共享component factory；保留projector/占位符/labels/shift练习，保留toy默认|
|OpenVLA|双视觉融合、词嵌入/LM logits、动作token映射|共享真实LM；视觉双分支仍toy；安全codec隔离特殊ID，不声称具备DINOv2+SigLIP预训练|
|BLIP-2|raw视觉features、Q-Former、冻结LM的embedding forward|共享组件可作为后续外围；Q-Former和多目标训练仍属于课程，不用原生VLM替代|
|InstructBLIP|raw视觉、instruction tokenizer、query-only跨注意力、冻结LM|需要保留两个文本入口、指令感知Q-Former与prefix规则；本次只记录适配面|
|Flamingo|逐LM层插入gated cross-attention、视觉记忆、多图可见性|仅提供final hidden或KV不够，需要额外decoder-layer adapter；当前backends不宣称支持Flamingo层间注入|
|ACT|多相机视觉tokens、状态与CVAE/action decoder|可复用视觉编码合同；没有语言需求；不引入VLM开销，原模型保持|
|Diffusion Policy|历史图像编码、状态、条件去噪器|可复用vision特征合同，时间U-Net/Transformer和噪声目标仍在课程；不把VLM接入设成必需|
|MAE|可训练patch encoder、mask恢复、像素重构|保留从零机制；基础模型加载最多作为后续初始化路线，不能跳过重构练习|
|DINO+iBOT|student/teacher多视图特征、heads、EMA、center|预训练视觉塔不能替代正在学习的训练目标/teacher更新；原填答不动|
|I-JEPA|visible-context编码、完整teacher与target gather|需要显式token可见性与teacher路径，不能仅调用通用VLM特征接口|
|V-JEPA|3D tubelet、视频网格、时空mask与teacher|当前2D vision接口不足以宣称视频兼容；未来需独立视频能力，不硬塞到图像接口|
|LDM_sd|已有CLIPTokenizer.from_pretrained入口；模型/训练为本地实现|保留现有tokenizer路线；不把文本tokenizer误称为预训练CLIP文本编码器，也不替换成VLM|
|其余生成模型与flow课程|自写架构与训练/sampling|没有必要新增VLM依赖。可共用低层工具，但动作flow的时间约定与图像flow保持各自明确|

## 接口规则

1. `VisionFeatures`显式标明mask/grid/CLS；区分raw与projected视觉特征，不能静默丢patch或假设CLS。
2. `language`支持embeddings、mask、positions和logits；课程负责loss及标签，loader不替课程定义监督。
3. `PrefixEncoding`保留每层K/V；不把单层hidden_states伪装成逐层action expert的输入。缓存只在同轮观测有效。
4. 冻结是参数策略，`no_grad`是计算图策略，二者不同。冻结语言模型时，输入projector仍可反传。
5. 训练与替身测试不能写回学生答案。新增可选组件通过工厂注入，旧demo和默认输出合同保持。
6. checkpoint按revision记录，下载与运行分开；无缓存时给出下载命令，不默默联网或回退tiny。

## 当前可复用的程度

三篇接入课程共享模型缓存和加载、视觉/文本权重访问、metadata、冻结与低层K/V合同；各自的projector、动作专家、量化、mask与训练循环仍可单独练习。没有为“统一”强改所有旧模型的参数布局，也不建立另一套扩散utils。

下一次扩展优先针对实际课程补明确能力：例如BLIP系只需embedding forward，Flamingo则需要经过测试的逐层注入；不要先承诺任意模型和任意课程互换。
