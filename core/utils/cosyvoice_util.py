import os.path
import sys

sys.path.append('third_party/Matcha-TTS')

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


class VoiceAPI:

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
        VoiceAPI.__voice_model = AutoModel(model=VoiceAPI.__voice_model_dir, device="cuda:0", disable_update=True)
        VoiceAPI.__cosyvoice_model = CosyVoice2(VoiceAPI.__cosyvoice_model_dir, load_jit=False, load_trt=False, fp16=True, use_flow_cache=False)


    @staticmethod
    def voice_to_text(wav_file_path: str) -> tuple[str, bool]:
        """
        语音转文字
        """
        try:
            res = VoiceAPI.__voice_model.generate(input=wav_file_path, cache={}, language="auto", use_itn=True)
            txt = rich_transcription_postprocess(res[0]["text"])
            return txt, True
        except Exception as e:
            print(e)
            return "", False
        

    @staticmethod
    def text_to_voice(device_id: str, text: str, wav_file_path: str) -> tuple[bool, float]:
        """
        文字转语音
        :param device_id: 设备id
        :param text: 文字
        :param wav_file_path: 保存路径
        :return: 是否成功, 时长
        """
        base_wav = f"voice/{device_id}/base.wav"
        base_wav_text = f"voice/{device_id}/base.txt"

        if not os.path.exists(base_wav) or not os.path.exists(base_wav_text):
            print(f"[text_to_voice] base wav does not exist: {base_wav} / ${base_wav_text}")
            return False, 0

        # 加载声音样本
        speech = load_wav(base_wav, 16000)
        with open(base_wav_text, 'r', encoding='utf-8') as f:
            wav_text = f.read()

        # 按样本合成
        for i, j in enumerate(VoiceAPI.__cosyvoice_model.inference_zero_shot(text, wav_text, speech, stream=False)):
            torchaudio.save(wav_file_path, j['tts_speech'], VoiceAPI.__cosyvoice_model.sample_rate)

        # 获取wav对象和采样率
        w, sr = torchaudio.load(wav_file_path)
        # 计算样本数量
        ns = w.size(1)
        # 计算时长
        duration = ns / sr

        return True, duration
