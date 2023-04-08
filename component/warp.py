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
    @staticmethod
    def merge_cell(result: List[dict]) -> List[dict]:
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
                    # 计算列表内文本长度
                    _length = sum([len(_c) for _c in _cache])
                    _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
                _cache = [_result["text"]]
                last_lang = _result["lang"]
        if _cache:
            _length = sum([len(_c) for _c in _cache])
            _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
        return _merged

    def create_cell(self,
                    sentence: str,
                    merge_same: bool = True,
                    cell_limit: int = 150,
                    filiter_space: bool = True) -> list:
        """
        分句，识别语言
        :param sentence: 句子
        :param merge_same: 是否合并相同语言的句子
        :param cell_limit: 单元最大长度
        :return:
        """
        cut = Cut()
        cut_list = cut.chinese_sentence_cut(sentence)
        _cut_list = []
        for _cut in cut_list:
            if len(_cut) > cell_limit:
                _text_list = [text[i:i + cell_limit] for i in range(0, len(text), cell_limit)]
                _cut_list.extend(_text_list)
            else:
                _cut_list.append(_cut)
        # 为每个句子标排语言
        _result = []
        for _cut in _cut_list:
            _lang = Detector.detect_language(_cut)
            if not filiter_space:
                _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
            else:
                if _lang:
                    _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
        if merge_same:
            _result = self.merge_cell(_result)
        return _result

    def build_sentence(self,
                       sentence_cell: List[dict],
                       strip: bool = False
                       ):
        # 生成句子
        _sentence = []
        for _cell in sentence_cell:
            _text = _cell.get('text')
            _lang = _cell.get('lang')
            if _lang:
                if strip:
                    _sentence.append(f"[{_lang}]{_text.strip()}[{_lang}]")
                else:
                    _sentence.append(f"[{_lang}]{_text}[{_lang}]")
        return _sentence

    def pack_up_task(self,
                     sentence_cell: List[dict],
                     task_limit: int = 150,
                     strip: bool = False
                     ):
        """
        打包单元
        :param sentence_cell: 单元列表
        :param task_limit: 任务最大长度
        :param strip: 是否去除空格
        :return:
        """
        _task_list = []
        _task = []
        _task_length = 0
        for _cell in sentence_cell:
            _text = _cell.get('text')
            _lang = _cell.get('lang')
            _length = _cell.get('length')
            if _lang:
                if _task_length + _length > task_limit:
                    _task_list.append(self.build_sentence(_task, strip=strip))
                    _task = []
                    _task_length = 0
                _task.append(_cell)
                _task_length += _length
        if _task:
            _task_list.append(self.build_sentence(_task, strip=strip))
        return _task_list


if __name__ == '__main__':
    text = """
    1. 今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天。翻译：Today is a sunny day with bright sunshine and fresh air. I plan to take a walk in the park and enjoy this beautiful day.
    
    2. 무엇인가를 생각하면 답답하거나 짜증나지 않고 미소 머금을 수 있는 하루였으면 좋겠습니다. 翻译：I hope to have a day where I can smile instead of feeling frustrated or annoyed when thinking about something.
    
    3. 早上好的韩文翻译是：짜증나지. 翻译：The Korean translation for "good morning" is "안녕하세요" (annyeonghaseyo).
    
    饮食男女，人之大欲也。性格天生，各有千秋。不可移易之物，爱恨情仇皆起于此。人生苦短，何必怀旧？前车之鉴，后事之师。日出而作，日落而息，勤奋正是成功之母。忍耐是一种美德，约束自己才能超越自己。道德是社会文明之基石，诚信是立身之本。文章合适，不在长短，在于内容。行路难，始知世间艰辛，但愿人长久，千里共婵娟。岁月匆匆，光阴如箭，唯有心存善念，方能始终如一。
    
    日本語が話せますので、何かお手伝いできることがありましたら、遠慮なくお申し付けください。日本には美しい自然と文化がたくさんあります。桜や紅葉の季節には多くの人々が訪れ、素晴らしい景色を楽しんでいます。また、日本の食文化も世界的に有名で、寿司やラーメン、うどんなど様々な料理があります。日本語は少し難しい言語かもしれませんが、練習することで上達します。一緒に頑張りましょう！
    """
    import time

    time1 = time.time()
    parse = Parse()
    sentence_cell = parse.create_cell(text, merge_same=False, cell_limit=140)
    print(sentence_cell)
    res = parse.build_sentence(sentence_cell=sentence_cell, strip=True)
    print(res)
    res2 = parse.pack_up_task(sentence_cell=sentence_cell, task_limit=140, strip=True)
    print(res2)
    print(res2[0])
    time2 = time.time()
    print(time2 - time1)
