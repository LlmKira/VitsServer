class VITS {
    constructor(base_url) {
        this.base_url = base_url;
    }

    async get_model_list(show_speaker = false, show_ms_config = false) {
        const url = `${this.base_url}/model/list?show_speaker=${show_speaker}&show_ms_config=${show_ms_config}`;
        const res = await fetch(url);
        return await res.json();
    }

    async get_model_info(model_id) {
        const url = `${this.base_url}/model/info?model_id=${model_id}`;
        const res = await fetch(url);
        return await res.json();
    }

    async parse_text(text, strip = false, merge_same = false, cell_limit = 140, filter_space = true) {
        const url = `${this.base_url}/tts/parse`;
        const data = {
            text,
            strip,
            merge_same,
            cell_limit,
            filter_space
        };
        const res = await fetch(url, {
            method: 'POST',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            }
        });
        return await res.json();
    }

    async generate_voice(model_id, text, speaker_id = 0, audio_type = "wav", length_scale = 1.0,
                         noise_scale = 0.5, noise_scale_w = 0.7, load_prefer = false, auto_parse = true) {
        let url = `${this.base_url}/tts/generate`;
        const data = {
            model_id,
            text,
            speaker_id,
            audio_type,
            length_scale,
            noise_scale,
            noise_scale_w,
            load_prefer
        };
        if (auto_parse) {
            url += "?auto_parse=True";
        }
        const res = await fetch(url, {
            method: 'POST',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            },
            responseType: 'blob'
        });
        return res.data;
    }
}

/*
const client = new VITS("http://0.0.0.0:9557");
const memory = await client.get_memory();
console.log(memory);

const modelList = await client.get_model_list(show_speaker=true, show_ms_config=true);
console.log(modelList);

const modelInfo = await client.get_model_info(model_id="model_01");
console.log(modelInfo);

const parsedText = await client.parse_text(text="Hello world!");
console.log(parsedText);

const voiceBlob = await client.generate_voice(model_id="model_01", text="[ZH]你好，世界！", speaker_id=0, audio_type="wav",
                            length_scale=1.0, noise_scale=0.5, noise_scale_w=0.6);
const voiceUrl = URL.createObjectURL(voiceBlob);
const audio = new Audio();
audio.src = voiceUrl;
audio.play();
* */