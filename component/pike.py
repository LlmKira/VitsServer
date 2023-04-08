# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 下午10:16
# @Author  : sudoskys
# @File    : pike.py
# @Software: PyCharm


class OnnxReader(object):

    @staticmethod
    def get_onnx_file_path(path: str):
        # 读取某个文件夹，将 .onnx 后缀的文件路径映射到一个kv表
        import os
        file_path = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == '.onnx':
                    file_path[os.path.splitext(file)[0]] = os.path.join(root, file)
        return file_path

    # 读取某个文件夹，如果有n个onnx 文件，就打包成 tar.gz
    @staticmethod
    def get_onnx_file(path: str, n: int = 5):
        """
        :param path: 文件夹路径
        :param n: onnx文件个数
        """
        import tarfile
        file_path = OnnxReader.get_onnx_file_path(path)
        if len(file_path) > n:
            tar = tarfile.open(path + ".tar.gz", "w:gz")
            for file in file_path:
                tar.add(file_path[file], arcname=file)
            tar.close()
            return path + ".tar.gz"
        else:
            return None
