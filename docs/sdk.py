import requests


class VITS:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_model_list(self, show_speaker=False, show_ms_config=False):
        url = f"{self.base_url}/model/list?show_speaker={show_speaker}&show_ms_config={show_ms_config}"
        res = requests.get(url)
        return res.json()

    def get_model_info(self, model_id):
        url = f"{self.base_url}/model/info?model_id={model_id}"
        res = requests.get(url)
        return res.json()

    def parse_text(self, text, strip=False, merge_same=False, cell_limit=140, filter_space=True):
        url = f"{self.base_url}/tts/parse"
        data = {
            "text": text,
            "strip": strip,
            "merge_same": merge_same,
            "cell_limit": cell_limit,
            "filter_space": filter_space
        }
        res = requests.post(url, json=data)
        return res.json()

    def generate_voice(self, model_id, text, speaker_id=0, audio_type="wav", length_scale=1.3,
                       noise_scale=0.6, noise_scale_w=0.6, load_prefer=False, auto_parse=True):
        url = f"{self.base_url}/tts/generate"
        data = {
            "model_id": model_id,
            "text": text,
            "speaker_id": speaker_id,
            "audio_type": audio_type,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "load_prefer": load_prefer
        }
        if auto_parse:
            url += "?auto_parse=True"
        res = requests.post(url, json=data, stream=True)
        return res


if __name__ == "__main__":
    client = VITS("http://127.0.0.1:9557")
    res = client.get_model_list(show_speaker=True, show_ms_config=True)
    print(res)

    res = client.get_model_info(model_id="model_01")
    print(res)

    res = client.parse_text(text="Hello world!")
    print(res)

    res = client.generate_voice(model_id="model_01", text="你好，世界！", speaker_id=0, audio_type="wav",
                                length_scale=1.0, noise_scale=0.0, noise_scale_w=0.0)
    with open("output.wav", "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
