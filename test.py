import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# 加载中文情感分析模型
sentiment_pipeline = pipeline("sentiment-analysis", model="uer/roberta-base-chinese-cluecorpusswwm-sentiment")
