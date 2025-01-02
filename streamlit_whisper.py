import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import tempfile


# 加载模型
st.title("Whisper App - Speech-to-Text")
st.text("Loading Whisper Model...")
model_path = "D:/github/Whisper-Finetune-master/models/whisper-tiny-finetune"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
st.success("Whisper Model Loaded")
# 上传音频文件
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
if audio_file:
    st.sidebar.header("Play Original Audio")
    st.sidebar.audio(audio_file)
# 点击按钮进行转录
if st.sidebar.button("Transcribe"):
    if audio_file is not None:
        st.sidebar.info("Processing Audio...")
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        # 加载音频
        audio, rate = librosa.load(tmp_file_path, sr=16000)
        # 转换为模型输入格式
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt").input_features
        # 推理
        with torch.no_grad():
            predicted_ids = model.generate(inputs)
        # 解码转录结果
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        # 显示结果
        st.sidebar.success("Transcription Completed!")
        st.markdown(f"### Transcription Result:\n\n{transcription}")
    else:
        st.sidebar.error("Please upload an audio file first.")



# streamlit run streamlit_whisper.py
