
# 使用指南:ASR与TTS实践

## 目录
1. [环境准备](#环境准备)
2. [TTS实践](#tts实践)
3. [ASR实践](#asr实践)
4. [高级应用](#高级应用)

---

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate soundfile librosa gradio

# 可选:加速库
pip install flash-attn --no-build-isolation
pip install liger-kernel

# 可选:音频处理
pip install pydub
```

### 模型下载

```python
# 使用 Hugging Face
from huggingface_hub import snapshot_download

# TTS模型
snapshot_download(
    repo_id="microsoft/VibeVoice-TTS",
    local_dir="./models/VibeVoice-TTS",
    local_dir_use_symlinks=False
)

# ASR模型
snapshot_download(
    repo_id="microsoft/VibeVoice-ASR",
    local_dir="./models/VibeVoice-ASR",
    local_dir_use_symlinks=False
)
```

---

## TTS实践

### 基础TTS推理

```python
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


class SimpleTTS:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2"
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading processor from {model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        print(f"Loading model from {model_path}")
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("TTS model loaded successfully!")
    
    def generate(
        self,
        text: str,
        voice_samples: list = None,
        cfg_scale: float = 1.3,
        inference_steps: int = 20,
        max_new_tokens: int = 1000
    ) -&gt; tuple:
        """
        生成语音
        
        Args:
            text: 要合成的文本
            voice_samples: 语音样本列表(文件路径或numpy数组)
            cfg_scale: CFG尺度
            inference_steps: 推理步数
            max_new_tokens: 最大生成token数
        
        Returns:
            (audio_array, sample_rate)
        """
        # 构造输入
        inputs = self.processor(
            text=text,
            voice_samples=voice_samples,
            return_tensors="pt"
        )
        
        # 移动到设备
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        # 设置推理步数
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False}
            )
        
        # 获取音频
        audio = outputs.speech_outputs[0].cpu().numpy()
        sample_rate = 24000
        
        return audio, sample_rate
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 24000):
        """保存音频到文件"""
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化
    tts = SimpleTTS(
        model_path="./models/VibeVoice-TTS",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 生成语音
    text = "Speaker 0: 你好,很高兴认识你! 这是一个测试。"
    
    # 如果有参考语音
    voice_samples = ["./voices/speaker0.wav"] if Path("./voices/speaker0.wav").exists() else None
    
    audio, sr = tts.generate(
        text=text,
        voice_samples=voice_samples,
        cfg_scale=1.3,
        inference_steps=20
    )
    
    # 保存
    tts.save_audio(audio, "output.wav")
```

### 流式TTS推理

```python
import torch
import numpy as np
import sounddevice as sd
from pathlib import Path
import threading
import queue

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.modular.streamer import AudioStreamer


class StreamingTTS:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        
        print(f"Loading processor from {model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        print(f"Loading streaming model from {model_path}")
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("Streaming TTS model loaded successfully!")
    
    def generate_streaming(
        self,
        text: str,
        voice_embeddings: list = None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        play_audio: bool = True
    ) -&gt; np.ndarray:
        """
        流式生成并可选播放
        
        Args:
            text: 要合成的文本
            voice_embeddings: 预计算的语音嵌入
            cfg_scale: CFG尺度
            inference_steps: 推理步数
            play_audio: 是否实时播放
        
        Returns:
            完整的音频数组
        """
        # 构造输入
        inputs = self.processor(
            text=text,
            return_tensors="pt"
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        # 设置推理步数
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)
        
        # 创建streamer
        streamer = AudioStreamer(batch_size=1)
        audio_queue = queue.Queue()
        
        # 播放线程
        def play_thread():
            buffer = []
            for audio_chunk in streamer.get_stream(0):
                chunk_np = audio_chunk.numpy()
                buffer.append(chunk_np)
                if play_audio:
                    sd.play(chunk_np, samplerate=24000)
                    sd.wait()
                audio_queue.put(chunk_np)
            
            audio_queue.put(None)  # 结束标记
        
        # 启动播放线程
        if play_audio:
            threading.Thread(target=play_thread, daemon=True).start()
        
        # 生成
        print("Generating and playing audio...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                audio_streamer=streamer
            )
        
        # 收集完整音频
        full_audio = []
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            full_audio.append(chunk)
        
        return np.concatenate(full_audio)


# 使用示例
if __name__ == "__main__":
    tts = StreamingTTS(
        model_path="./models/VibeVoice-Streaming-TTS",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    text = "Speaker 0: 这是一个流式TTS的示例。你可以实时听到生成的语音!"
    
    audio = tts.generate_streaming(
        text=text,
        cfg_scale=1.3,
        inference_steps=10,
        play_audio=True
    )
    
    import soundfile as sf
    sf.write("streaming_output.wav", audio, 24000)
```

### 多说话人对话生成

```python
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


class MultiSpeakerTTS:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
        
        # 存储说话人样本
        self.speaker_voices = {}
    
    def register_speaker(self, speaker_id: str, voice_sample: str):
        """注册一个说话人的语音样本"""
        self.speaker_voices[speaker_id] = voice_sample
        print(f"Registered speaker: {speaker_id}")
    
    def generate_dialogue(
        self,
        dialogue_script: str,
        cfg_scale: float = 1.3,
        inference_steps: int = 20
    ) -&gt; tuple:
        """
        生成多人对话
        
        脚本格式:
            Speaker 0: 你好!
            Speaker 1: 你好,很高兴认识你!
        """
        # 收集所有使用的说话人
        import re
        speaker_ids = set()
        for line in dialogue_script.strip().split("\n"):
            match = re.match(r"Speaker\s+(\d+):", line)
            if match:
                speaker_ids.add(match.group(1))
        
        # 获取语音样本
        voice_samples = []
        for sid in sorted(speaker_ids, key=int):
            if sid in self.speaker_voices:
                voice_samples.append(self.speaker_voices[sid])
            else:
                print(f"Warning: No voice sample for speaker {sid}, using default")
                # 可以在这里添加默认语音
        
        # 构造输入
        inputs = self.processor(
            text=dialogue_script,
            voice_samples=voice_samples if voice_samples else None,
            return_tensors="pt"
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer
            )
        
        return outputs.speech_outputs[0].cpu().numpy(), 24000


# 使用示例
if __name__ == "__main__":
    tts = MultiSpeakerTTS(
        model_path="./models/VibeVoice-TTS",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 注册说话人
    if Path("./voices/speaker0.wav").exists():
        tts.register_speaker("0", "./voices/speaker0.wav")
    if Path("./voices/speaker1.wav").exists():
        tts.register_speaker("1", "./voices/speaker1.wav")
    
    # 对话脚本
    dialogue = """
    Speaker 0: 你好,欢迎使用VibeVoice!
    Speaker 1: 谢谢你,这个系统真不错!
    Speaker 0: 是的,它支持多人对话生成。
    Speaker 1: 那我们来演示一段对话吧。
    """
    
    audio, sr = tts.generate_dialogue(dialogue, cfg_scale=1.3)
    sf.write("dialogue_output.wav", audio, sr)
```

---

## ASR实践

### 基础ASR推理

```python
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import json

from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration


class SimpleASR:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2"
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading ASR processor from {model_path}")
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        
        print(f"Loading ASR model from {model_path}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("ASR model loaded successfully!")
    
    def transcribe(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        context_info: str = None
    ) -&gt; dict:
        """
        识别音频到文本
        
        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组(替代audio_path)
            sample_rate: 采样率
            max_new_tokens: 最大生成token数
            temperature: 温度
            context_info: 上下文信息(可选)
        
        Returns:
            {
                "raw_text": 原始文本,
                "segments": [
                    {
                        "start_time": 开始时间,
                        "end_time": 结束时间,
                        "speaker_id": 说话人ID,
                        "text": 文本
                    },
                    ...
                ]
            }
        """
        # 处理音频
        inputs = self.processor(
            audio=audio_path if audio_path is not None else audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        
        # 移动到设备
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        # 生成配置
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature &gt; 0 else None,
            "do_sample": temperature &gt; 0,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)
        
        # 解码
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # 解析结构化输出
        segments = self._parse_segments(raw_text)
        
        return {
            "raw_text": raw_text,
            "segments": segments
        }
    
    def _parse_segments(self, raw_text: str) -&gt; list:
        """解析模型输出的结构化片段"""
        segments = []
        try:
            # 尝试解析JSON格式
            import re
            # 提取JSON数组
            json_match = re.search(r'\[(.*)\]', raw_text, re.DOTALL)
            if json_match:
                json_str = "[" + json_match.group(1) + "]"
                import json
                segments = json.loads(json_str)
        except:
            pass
        
        return segments


# 使用示例
if __name__ == "__main__":
    asr = SimpleASR(
        model_path="./models/VibeVoice-ASR",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 识别文件
    if Path("./test_audio.wav").exists():
        result = asr.transcribe(
            audio_path="./test_audio.wav",
            max_new_tokens=1024,
            context_info="会议记录 人工智能"
        )
        
        print("Raw output:", result["raw_text"])
        print("\nSegments:")
        for i, seg in enumerate(result["segments"]):
            print(f"{i+1}. [{seg.get('start_time')}-{seg.get('end_time')}] Speaker {seg.get('speaker_id')}: {seg.get('text')}")
        
        # 保存结果
        with open("transcription.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
```

### 流式ASR推理

```python
import torch
import numpy as np
import sounddevice as sd
import queue
import threading
from pathlib import Path
import time

from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from transformers import TextIteratorStreamer


class StreamingASR:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
        
        # 音频缓冲
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.sample_rate = 24000
    
    def _record_audio(self, duration: float = 5.0):
        """录制音频"""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        print("Recording complete!")
        return audio.flatten()
    
    def transcribe_streaming(
        self,
        audio_path: str = None,
        duration: float = 10.0,
        max_new_tokens: int = 1024
    ):
        """
        流式识别
        
        Args:
            audio_path: 音频文件(或None以录音)
            duration: 录音时长
        """
        # 获取音频
        if audio_path and Path(audio_path).exists():
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        else:
            audio = self._record_audio(duration)
            sr = self.sample_rate
        
        # 创建streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 处理音频
        inputs = self.processor(
            audio_array=audio,
            sampling_rate=sr,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        # 在后台线程中运行
        def run_generate():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    pad_token_id=self.processor.pad_id
                )
        
        threading.Thread(target=run_generate, daemon=True).start()
        
        # 流式输出
        print("\nTranscription:")
        print("-" * 50)
        full_text = ""
        for new_text in streamer:
            full_text += new_text
            print(new_text, end="", flush=True)
        
        print("\n" + "-" * 50)
        print("\nComplete!")
        
        return full_text


# 使用示例
if __name__ == "__main__":
    asr = StreamingASR(
        model_path="./models/VibeVoice-ASR",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 从文件识别
    if Path("./test_audio.wav").exists():
        asr.transcribe_streaming("./test_audio.wav")
    else:
        # 录制并识别
        asr.transcribe_streaming(duration=10)
```

---

## 高级应用

### ASR-TTS双向应用:语音对话系统

```python
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import queue
import threading

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration


class VoiceAssistant:
    def __init__(
        self,
        tts_model_path: str,
        asr_model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.sample_rate = 24000
        
        print("Loading TTS model...")
        self.tts_processor = VibeVoiceProcessor.from_pretrained(tts_model_path)
        self.tts_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            tts_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.tts_model.eval()
        
        print("Loading ASR model...")
        self.asr_processor = VibeVoiceASRProcessor.from_pretrained(asr_model_path)
        self.asr_model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            asr_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.asr_model.eval()
        
        print("Voice assistant ready!")
        
        # 对话历史
        self.dialogue_history = []
    
    def listen(self, duration: float = 5.0) -&gt; str:
        """录音并识别"""
        print(f"\nListening for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        
        print("Processing...")
        inputs = self.asr_processor(
            audio_array=audio.flatten(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        with torch.no_grad():
            output_ids = self.asr_model.generate(
                **inputs,
                max_new_tokens=512
            )
        
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.asr_processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"You said: {text}")
        return text
    
    def speak(self, text: str):
        """合成并播放语音"""
        print(f"Speaking: {text}")
        
        input_text = f"Speaker 0: {text}"
        
        inputs = self.tts_processor(
            text=input_text,
            return_tensors="pt"
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        self.tts_model.set_ddpm_inference_steps(num_steps=10)
        
        with torch.no_grad():
            outputs = self.tts_model.generate(
                **inputs,
                max_new_tokens=500,
                cfg_scale=1.3,
                tokenizer=self.tts_processor.tokenizer
            )
        
        audio = outputs.speech_outputs[0].cpu().numpy()
        
        # 播放
        sd.play(audio, samplerate=self.sample_rate)
        sd.wait()
    
    def chat(self):
        """简单的对话循环"""
        print("=" * 50)
        print("Voice Assistant Ready!")
        print("Say 'quit' or 'exit' to end.")
        print("=" * 50)
        
        while True:
            user_text = self.listen(duration=5.0)
            
            if any(keyword in user_text.lower() for keyword in ["quit", "exit", "结束", "退出"]):
                print("Goodbye!")
                self.speak("再见!")
                break
            
            # 简单的回复逻辑
            response = self._generate_response(user_text)
            
            self.speak(response)
            
            # 保存历史
            self.dialogue_history.append({"user": user_text, "assistant": response})
    
    def _generate_response(self, user_text: str) -&gt; str:
        """简单的回复生成(可替换为LLM)"""
        user_text_lower = user_text.lower()
        
        if "你好" in user_text_lower or "hello" in user_text_lower:
            return "你好! 很高兴见到你。"
        elif "谢谢" in user_text_lower or "thank" in user_text_lower:
            return "不客气! 还有什么可以帮你的吗?"
        elif "天气" in user_text_lower:
            return "今天天气很不错,适合出去散步。"
        elif "名字" in user_text_lower:
            return "我是VibeVoice助手,很高兴为你服务。"
        else:
            return f"我听到了: {user_text}。这是一个简单的演示。"


# 使用示例
if __name__ == "__main__":
    assistant = VoiceAssistant(
        tts_model_path="./models/VibeVoice-TTS",
        asr_model_path="./models/VibeVoice-ASR",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    assistant.chat()
```

### 批量处理

```python
import torch
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from vibevoice.processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


class BatchProcessor:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        voice_sample: str = None,
        cfg_scale: float = 1.3,
        inference_steps: int = 20
    ):
        """
        批量处理目录下的文本文件
        
        文件格式: .txt文件,每行一个句子
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        text_files = list(input_path.glob("*.txt"))
        print(f"Found {len(text_files)} text files")
        
        for txt_file in tqdm(text_files, desc="Processing"):
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            
            # 生成
            audio, sr = self._generate_single(
                text=text,
                voice_sample=voice_sample,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps
            )
            
            # 保存
            output_file = output_path / f"{txt_file.stem}.wav"
            sf.write(output_file, audio, sr)
        
        print("Batch processing complete!")
    
    def _generate_single(
        self, text: str, voice_sample: str = None, cfg_scale: float = 1.3, inference_steps: int = 20
    ) -&gt; tuple:
        inputs = self.processor(
            text=f"Speaker 0: {text}",
            voice_samples=[voice_sample] if voice_sample else None,
            return_tensors="pt"
        )
        
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer
            )
        
        return outputs.speech_outputs[0].cpu().numpy(), 24000


# 使用示例
if __name__ == "__main__":
    processor = BatchProcessor(
        model_path="./models/VibeVoice-TTS",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    processor.process_directory(
        input_dir="./input_texts",
        output_dir="./output_audios",
        voice_sample="./voices/speaker0.wav"
    )
```

---

## 总结

本指南展示了VibeVoice的多种使用方式:

1. **TTS**:基础TTS、流式TTS、多说话人对话
2. **ASR**:基础ASR、流式ASR
3. **高级应用**:语音对话系统、批量处理

关键要点:
- 使用`flash_attention_2`可以加速推理
- `cfg_scale`在1.0-2.0之间通常有较好效果
- 推理步数越少越快,但质量可能下降
- 支持多种音频格式和采样率

更多示例请参考项目中的demo目录!
