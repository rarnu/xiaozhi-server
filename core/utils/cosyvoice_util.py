import io
import os.path
import sys

sys.path.append('third_party/Matcha-TTS')

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


class CosyVoiceAPI:

    # modelscope download --model iic/SenseVoiceSmall --local_dir ./iic/SenseVoiceSmall
    # modelscope download --model iic/CosyVoice2-0.5B --local_dir ./iic/CosyVoice2-0.5B
    # modelscope download --model iic/CosyVoice-ttsfrd --local_dir ./iic/CosyVoice-ttsfrd
    # cd CosyVoice-ttsfrd
    # unzip resource.zip -d .
    # pip install ttsfrd_dependency-0.1-py3-none-any.whl
    # pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

    __voice_model_dir = "iic/SenseVoiceSmall"
    __cosyvoice_model_dir = "iic/CosyVoice2-0.5B"
    __voice_model: AutoModel | None
    __cosyvoice_model: CosyVoice2 | None


    @staticmethod
    def init_voice():
        CosyVoiceAPI.__voice_model = AutoModel(model=CosyVoiceAPI.__voice_model_dir, device="cuda:0", disable_update=True)
        CosyVoiceAPI.__cosyvoice_model = CosyVoice2(CosyVoiceAPI.__cosyvoice_model_dir, load_jit=False, load_trt=False, fp16=True, use_flow_cache=False)


    @staticmethod
    def voice_to_text(wav_file_path: str) -> tuple[str, bool]:
        """
        语音转文字
        """
        try:
            res = CosyVoiceAPI.__voice_model.generate(input=wav_file_path, cache={}, language="auto", use_itn=True)
            txt = rich_transcription_postprocess(res[0]["text"])
            return txt, True
        except Exception as e:
            print(e)
            return "", False
        

    @staticmethod
    def text_to_voice(sample_wav_path: str, sample_text: str, text: str, wav_file_path: str | None = None) -> bytes | None:
        
        try:
            # 加载声音样本
            speech = load_wav(sample_wav_path, 16000)

            buffer = io.BytesIO()

            # 按样本合成
            for i, j in enumerate(CosyVoiceAPI.__cosyvoice_model.inference_zero_shot(text, sample_text, speech, stream=False)):
                if wav_file_path:
                    torchaudio.save(wav_file_path, j['tts_speech'], CosyVoiceAPI.__cosyvoice_model.sample_rate)
                else:
                    torchaudio.save(buffer, j['tts_speech'], CosyVoiceAPI.__cosyvoice_model.sample_rate, format="wav")

            if wav_file_path:
                # 从 wav 读取
                with open(wav_file_path, 'rb') as f:
                    buffer = f.read()
                return buffer
            else:
                return buffer.getvalue()
        except Exception as e:
            print(f"语音合成失败: {str(e)}")
            return None
