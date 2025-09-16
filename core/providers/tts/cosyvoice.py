import os
import uuid
from datetime import datetime
from core.providers.tts.base import TTSProviderBase
from core.utils.cosyvoice_util import CosyVoiceAPI

class TTSProvider(TTSProviderBase):

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.voice_demo = config.get("voice_demo")
        self.voice_remark = config.get("voice_remark")


    def generate_filename(self, extension=".wav"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    async def text_to_speak(self, text, output_file):
        print(f"开始合成音频: {output_file}")
        try:
            sample_wav = os.path.join('sound', self.voice_demo)
            return CosyVoiceAPI.text_to_voice(sample_wav, self.voice_remark, text, output_file)
        except Exception as e:
            error_msg = f"XJ-CosyVoice TTS请求失败: {e}"
            raise Exception(error_msg)  # 抛出异常，让调用方捕获