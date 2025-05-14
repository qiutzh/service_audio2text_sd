## 语音识别

- [Paraformer-large热词版模型](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)

### 工作进展20250428

1）说话者分离
  a.paraformer普通版和热词版，说话者分离对比试验；
    怎么新增热词，同音不同字热词怎么添加权重；
  b.sensevoice说话者分离cpu版本，验证效果；
    gpu版本进行中（有个技术难点待解决）
c.

2）微调
调研结果：有开发者做过测试，sensevoice比paraformer-large准确度高2%左右（88%->90%）
1）sensevoice微调（下一步工作重心）
  a.标注样本格式，准备一小批微调样本；
  b.打通sensevoice微调流程；
  c.找到一个有效的语音标注工具；

**参考**
- [微调数据格式](https://github.com/modelscope/FunASR/tree/main/data/list)
- [微调脚本](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/finetune.sh)
- [微调部署教程：微调后调用需要注册](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md)
- [sensevoice微调官方教程](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/sense_voice/README_zh.md)
- [performer微调官方教程](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README_zh.md)
- [performer热词版微调脚本](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/seaco_paraformer/finetune.sh)
- [sensevoice方言微调注意事项](https://github.com/FunAudioLLM/SenseVoice/issues/58)

**问题**

问题：SeACo中的热词模块中的热词的先后顺序有要求吗？对于转录的每个字都需要遍历整个热词列表吗？
解答：没有要求，先后顺序不影响热词激励。热词激励的原理不是遍历热词列表进行匹配的方式，可以阅读模型代码

- [Paraformer-large热词版模型-说话人识别不准](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/feedback/issueDetail/28261)

## 说话者分离

### 开源工具

- [sherpa-onnx：打包好的onnx模型直接拿来用【全部是cpu推理】](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)


## 公开数据集

- [中文普通话语音识别开源数据集（持续更新）](https://blog.csdn.net/chan1987818/article/details/108746112)
- [好未来开放语音数据集](https://www.geekpark.net/news/274193); [下载链接](https://ai.100tal.com/openData/voice)