# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 下午10:13
# @Author  : sudoskys
# @File    : warp.py
# @Software: PyCharm
from typing import List

from component.nlp_utils.cut import Cut
from component.nlp_utils.detect import DetectSentence

Detector = DetectSentence()


class Parse(object):
    def merge(self, result: List[dict]) -> List[dict]:
        """
        合并句子
        :param result:
        :return:
        """
        _merged = []
        _cache = []
        last_lang = None
        for _result in result:
            if _result["lang"] == last_lang:
                _cache.append(_result["text"])
            else:
                if _cache:
                    _merged.append({"text": "".join(_cache), "lang": last_lang})
                _cache = [_result["text"]]
                last_lang = _result["lang"]
        if _cache:
            _merged.append({"text": "".join(_cache), "lang": last_lang})
        return _merged

    def warp_sentence(self, text: str):
        """
        分句，识别语言
        :param text:
        :return:
        """
        cut = Cut()
        _cut_list = cut.chinese_sentence_cut(text)
        # 为每个句子标排语言
        _text = []
        for _cut in _cut_list:
            _lang = Detector.detect_language(_cut)
            _text.append({"text": _cut, "lang": _lang})
        _text = self.merge(_text)
        return _text

    def build_vits_sentence(self, merged: List[dict], strip: bool = False):
        """
        [{'text': '', 'lang': ''}, {'text': '    1. 今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天。翻译：', 'lang': 'ZH'},]
        :param merged: 合并后的句子
        :return: 用于 VITS 的句子
        """
        # 用Lang标签包裹句子
        _sentence = []
        for _merged in merged:
            _lang = _merged.get('lang')
            _text: str = _merged.get('text')
            if _lang:
                # 如果文本不是空格则添加
                if _text.isspace():
                    _sentence.append(_text)
                else:
                    if strip:
                        _sentence.append(f"[{_lang}]{_text.strip()}[{_lang}]")
                    else:
                        _sentence.append(f"[{_lang}]{_text}[{_lang}]")
        return " ".join(_sentence)


if __name__ == '__main__':
    text = """
    1. 今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天。翻译：Today is a sunny day with bright sunshine and fresh air. I plan to take a walk in the park and enjoy this beautiful day.
    
    2. 무엇인가를 생각하면 답답하거나 짜증나지 않고 미소 머금을 수 있는 하루였으면 좋겠습니다. 翻译：I hope to have a day where I can smile instead of feeling frustrated or annoyed when thinking about something.
    
    3. 早上好的韩文翻译是：짜증나지. 翻译：The Korean translation for "good morning" is "안녕하세요" (annyeonghaseyo).

    """
    import time

    time1 = time.time()
    parse = Parse()
    res = (parse.warp_sentence(text))
    res2 = parse.build_vits_sentence(res, strip=True)
    print(res2)
    time2 = time.time()
    print(time2 - time1)
