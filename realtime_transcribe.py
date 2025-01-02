import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# 加载微调后的模型和处理器
MODEL_PATH = "D:/github/Whisper-Finetune-master/models/whisper-tiny-finetune"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
model.eval()

# 设置强制解码器 ID
forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="zh", task="transcribe")
print("forced_decoder_ids:", forced_decoder_ids)
# 音频参数
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # 每次处理 2 秒音频
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
# 实时音频处理函数
def audio_callback(indata, frames, time, status):
    if status:
        print("音频状态错误:", status)
    # 将音频转换为 Whisper 格式
    audio_chunk = indata[:, 0]
    audio_chunk = audio_chunk.astype(np.float32)
    audio_chunk /= np.max(np.abs(audio_chunk))  # 归一化到 [-1, 1]
    # 模型推理
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids,
            repetition_penalty=2.0,
            max_length=30,
            temperature=0.7,
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # 限制输出长度为 15 个字
    limited_transcription = transcription[:20]
    print(limited_transcription)

# 启动实时录音
print("开始实时转录...按 Ctrl+C 停止")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE
    ):
        sd.sleep(10000000)
except KeyboardInterrupt:
    print("实时转录已停止")
except Exception as e:
    print(f"发生错误: {e}")
