# -*- coding: utf-8 -*-
from loguru import logger


class DetectSentence(object):
    """
    检测句子
    """

    @staticmethod
    def detect_language(sentence: str):
        """
        Detect language
        :param sentence: sentence
        :return: 两位大写语言代码 (EN, ZH, JA, KO, FR, DE, ES, ....)
        """
        # 如果全部是空格
        if sentence.isspace() or not sentence:
            return ""

        # 如果全部是标点
        try:
            from .. import langdetect_fasttext
            lang_type = langdetect_fasttext.detect(text=sentence.replace("\n", "").replace("\r", ""),
                                                   low_memory=True).get("lang").upper()

            def is_japanese(string):
                for ch in string:
                    if 0x3040 < ord(ch) < 0x30FF:
                        return True
                return False

            if lang_type == "JA" and not is_japanese(sentence):
                lang_type = "ZH"
        except Exception as e:
            # handle error
            logger.error(e)
            raise e
        return lang_type

    @staticmethod
    def detect_help(sentence: str) -> bool:
        """
        检测是否是包含帮助要求，如果是，返回True，否则返回False
        """
        _check = ['怎么做', 'How', 'how', 'what', 'What', 'Why', 'why', '复述', '复读', '要求你', '原样', '例子',
                  '解释', 'exp', '推荐', '说出', '写出', '如何实现', '代码', '写', 'give', 'Give',
                  '请把', '请给', '请写', 'help', 'Help', '写一', 'code', '如何做', '帮我', '帮助我', '请给我', '什么',
                  '为何', '给建议', '给我', '给我一些', '请教', '建议', '怎样', '如何', '怎么样',
                  '为什么', '帮朋友', '怎么', '需要什么', '注意什么', '怎么办', '助け', '何を', 'なぜ', '教えて', '提案',
                  '何が', '何に',
                  '何をす', '怎麼做', '複述', '復讀', '原樣', '解釋', '推薦', '說出', '寫出', '如何實現', '代碼', '寫',
                  '請把', '請給', '請寫', '寫一', '幫我', '幫助我', '請給我', '什麼', '為何', '給建議', '給我',
                  '給我一些', '請教', '建議', '步驟', '怎樣', '怎麼樣', '為什麼', '幫朋友', '怎麼', '需要什麼',
                  '註意什麼', '怎麼辦']
        for item in _check:
            if item in sentence:
                return True
        return False

    @staticmethod
    def detect_code(sentence) -> bool:
        """
        Detect code，if code return True，else return False
        :param sentence: sentence
        :return: bool
        """
        code = False
        _reco = [
            '("',
            '")',
            ").",
            "()",
            "!=",
            "==",
        ]
        _t = len(_reco)
        _r = 0
        for i in _reco:
            if i in sentence:
                _r += 1
        if _r > _t / 2:
            code = True
        rms = [
            "```",
            "import "
            "print_r(",
            "var_dump(",
            'NSLog( @',
            'println(',
            '.log(',
            'print(',
            'printf(',
            'WriteLine(',
            '.Println(',
            '.Write(',
            'alert(',
            'echo(',
        ]
        for i in rms:
            if i in sentence:
                code = True
        return code
