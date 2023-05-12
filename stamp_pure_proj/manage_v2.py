# -*- coding: UTF-8 -*-

import sys
import os

sys.path.append(os.path.abspath("yolov5_6_1"))
sys.path.append(os.path.abspath("yolov5_obb_crease"))

from mysql_about.exts import db
from flask_sqlalchemy import SQLAlchemy

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

import json
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import flask_config

from adjust_common_v8 import adjust_single_img, adjust_perf_img
from pattern_pos.pat_pos_adjust import adjust_pat_img
# from adjust_perf import adjust_perf_img
import cut_margin
from detect_all_v8 import detectFront, detectBack
# from final_grade.grade_stamp_formula import cal_final_with_formula
from final_grade.grade_stamp_12 import grade_total_score
from QtProject import predict_0720

use_zh = True
eva_path = "./save_anchored_img_dir"
# 标准库文件在服务器的本地路径,需要修改
# stb_path = os.path.abspath('../standardlib')
stb_path = os.path.abspath('../dataset-downloads/standardlib')
assert os.path.exists(stb_path), "标准库路径导入错误"


class Config(object):
    """配置参数"""
    MYSQL_DIALECT = 'mysql'  # 使用哪个数据库
    MYSQL_DIRVER = 'pymysql'  # 选择驱动
    MYSQL_NAME = 'root'  # 用户名
    MYSQL_PWD = 'root'  # 密码
    MYSQL_HOST = 'localhost'  # 主机名
    MYSQL_PORT = 3306  # 端口号
    MYSQL_DB = 'StampImage'  # 数据库名
    MYSQL_CHARSET = 'utf8mb4'  # 编码格式

    SQLALCHEMY_DATABASE_URI = f'{MYSQL_DIALECT}+{MYSQL_DIRVER}://{MYSQL_NAME}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset={MYSQL_CHARSET}'
    # 默认设置为true，当数据发生变化，会发送一个信号。
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    # 设置加密字符
    SECRET_KEY = os.urandom(16)
    DEBUG = False  # Flask内置了调试模式，可以自动重载代码并显示调试信息

    # # 设置sqlalchemy自动更跟踪数据库
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 查询时会显示原始SQL语句
    SQLALCHEMY_ECHO = True
    # 禁止自动提交数据处理
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    # app.config['JSON_AS_ASCII'] = False  # 解决flask接口中文数据编码问题

    # 数据编码
    JSON_AS_ASCII = False


class StampImage(db.Model):
    # 定义表名
    __tablename__ = 'stampImages'

    # 定义字段
    data_id = db.Column(db.Integer, primary_key=True)
    front = db.Column(db.String(64))
    back = db.Column(db.String(64))
    front_com = db.Column(db.String(64))
    back_com = db.Column(db.String(64))
    front_perf = db.Column(db.String(64))
    perf_dir = db.Column(db.String(64))
    front_pos = db.Column(db.String(64))
    front_labelled = db.Column(db.String(64))
    back_labelled = db.Column(db.String(64))
    mat_com_b = db.Column(db.PickleType)
    cls_name = db.Column(db.String(32))

    # repr()方法显示一个可读字符串，不是完全必要，可用于调试和测试。
    def __repr__(self):
        return '<image data_id {}> '.format(self.data_id)

    # def __init__(self):
    #     self.data_id = data_id
    #     self.front = front
    #     self.back = back


def create_table():
    # # 删除表
    # db.drop_all()
    # 创建表
    db.create_all()


# # 判断表是否存在
# def table_exists(engine, table_name):
#     pass
#     # # tables = engine.session.
#     # for table_col in tables:
#     #     if table_name == table_col[0]:
#     #         return True
#     # return False
#     # except Exception as e:
#     #     print("Exception", str(e))
#     #     return "error"


# 读取配置
app = Flask(__name__)
app.config.from_object(Config)

# 设置可跨域范围
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = 'uploaded'
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

db = SQLAlchemy(app)
# connect_res = table_exists(db, "StampImage")


def allowed_file(filename):
    """
    判断文件类型是否允许上传
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'stamp project'


@app.route("/cls_up/<imageId>")
def get_frame_cls(imageId):
    # 图片上传保存的路径
    img_path = os.path.join(stb_path, imageId)
    with open(img_path, 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image")
        # resp = Response(image, mimetype="image/jpg")
        return resp


@app.route("/eva_up/<imageId>")
def get_frame_eva(imageId):
    # 图片上传保存的路径
    img_path = os.path.join(eva_path, imageId)
    with open(img_path, 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image")
        # resp = Response(image, mimetype="image/jpg")
        return resp


# 上传图片 执行前提：图片后缀，只允许jpg格式
# 输入：正面或背面图片，不接受正面、背面图同时输入，只允许一张一张输入
@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    前端传form-data
    传2张图片，参数名"front_img"、"back_img"
    """
    print("running upload()\n#################################################################")
    data_id = str(uuid.uuid4())
    print("upload() data_id", data_id, flush=True)

    # 改用 form data
    front_f = request.files['front_img']
    back_f = request.files['back_img']

    if not front_f and not back_f:
        res = {"errorCode": "100003", "errorMessage": "正面和背面图片已损坏", "data_id": data_id}
        return res
    elif not front_f:
        res = {"errorCode": "100001", "errorMessage": "正面图片已损坏", "data_id": data_id}
        return res
    elif not back_f:
        res = {"errorCode": "100002", "errorMessage": "背面图片已损坏", "data_id": data_id}
        return res

    front_secure_name = secure_filename(front_f.filename)
    print("upload() front_secure_name", front_secure_name, flush=True)
    front_ext = front_secure_name.rsplit('.')[-1]
    front_new_name = str(uuid.uuid4()) + "." + front_ext
    front_rel = os.path.join(app.config['UPLOAD_FOLDER'], front_new_name)
    front = os.path.abspath(front_rel).replace("\\", "/")
    front_f.save(front)

    # print("upload() front", front, flush=True)
    front_com, mat_com, front_perf, front_pos, perf_dir = pre_handle_img(front, "front")
    print("upload() mat_com", mat_com, flush=True)
    back_secure_name = secure_filename(back_f.filename)
    back_ext = back_secure_name.rsplit('.')[-1]
    back_new_name = str(uuid.uuid4()) + "." + back_ext
    back_rel = os.path.join(app.config['UPLOAD_FOLDER'], back_new_name)
    back = os.path.abspath(back_rel).replace("\\", "/")
    back_f.save(back)
    # back_img = np.fromstring(back_b, np.uint8)
    print("upload() front, back", front, back, flush=True)
    back_com = pre_handle_img(back, "back")

    # mat_com 改一下
    mat_com_b = mat_com.toBytes()
    tmp_stamp = StampImage(data_id=data_id,
                           front=front,
                           front_com=front_com,
                           front_perf=front_perf,
                           front_pos=front_pos,
                           perf_dir=perf_dir,
                           mat_com_b=mat_com_b,
                           back=back,
                           back_com=back_com
                           )
    db.session.add(tmp_stamp)
    db.session.commit()

    res = {"errorCode": "0", "errorMessage": "OK", "data_id": data_id}
    return jsonify(res)


# 预处理
def pre_handle_img(file_abs, side):
    if side == "front":
        front_com, mat_com = adjust_single_img(file_abs)
        front_perf, _ = adjust_perf_img(file_abs)
        front_perf = front_perf.replace("\\", "/")
        perf_dir = cut_margin.cut_perf_adj_margin(front_com,
                                                  front_perf).replace("\\", "/")
        front_pos = adjust_pat_img(file_abs).replace("\\", "/")
        return front_com, mat_com, front_perf, front_pos, perf_dir

    if side == "back":
        back_com, _ = adjust_single_img(file_abs)
        back_com = back_com.replace("\\", "/")
        return back_com


# 分类系统 执行前提：正面图片已输入
# 输入：data_id
@app.route("/classification", methods=["GET", "POST"])
def cls_stamp():
    print("running cls_stamp()\n#################################################################")
    data_json = request.get_data()
    # print("cls_stamp() data_json", data_json, flush=True)
    # print("cls_stamp() data_json.decode()", data_json.decode(), flush=True)
    data = json.loads(data_json.decode().replace("'", "\""))
    # print("cls_stamp() data", data, flush=True)
    data_id_get = data["data_id"]
    print("cls_stamp() data_id_get =", data_id_get, flush=True)
    stamp_queried = StampImage.query.filter(StampImage.data_id == data_id_get).first()
    # assert len(stamp_queried) == 1, "len(stamp_queried) != 1, data_id " + data_id_get + "重复"
    # if len(stamp_queried) == 0:
    if not stamp_queried:
        res = {"errorCode": "200012", "errorMessage": "数据库中不存在 data_id 使 data_id_get == data_id"}
        return res

    data_id = stamp_queried.data_id
    front = stamp_queried.front
    back = stamp_queried.back
    print("cls_stamp() data_id", data_id, flush=True)
    print("cls_stamp() front, back", front, back, flush=True)

    if not front:
        res = {"errorCode": "2000011", "errorMessage": "数据库未找到用户输入图片的保存路径"}
        return jsonify(res)

    cls_name, cls_prob = predict_0720.main(front)

    json_cls_name_path = r"class_indices.json"
    with open(json_cls_name_path, 'r', encoding='utf-8') as fp:
        dict_cls = json.load(fp)

    if not (cls_name in dict_cls):
        cls_name_zh = cls_name + ".jpg"
    cls_name_zh = dict_cls[cls_name]

    cls_result = os.path.join(stb_path, cls_name_zh)
    cls_result = os.path.abspath(cls_result).replace("\\", "/")

    cls_prob_thres = 0.85
    if cls_prob > cls_prob_thres and os.path.exists(cls_result):
        if "T46.1-1" in cls_name:
            cls_str = "分类结果: " + cls_name + "(庚申猴票)"
        else:
            cls_str = "分类结果: " + cls_name
    else:
        cls_str = "没有找到对应图片,邮票库里暂无该类邮票"
        res = {"errorCode": "200002", "errorMessage": cls_str}
        return jsonify(res)
    prefix = "http://" + flask_config.HOST + ":" + flask_config.PORT

    # 传url路径
    cls_venv_path = os.path.join(prefix, "cls_up", cls_name_zh).replace("\\", "/")
    print("cls_venv_path", cls_venv_path)
    # 返回：cls_str用于显示的分类结果 cls_img_path服务器需要显示的该类型图片在标准库的本地路径
    res = {"errorCode": "0",
           "errorMessage": "OK",
           "cls_str": cls_str,
           "cls_venv_path": cls_venv_path}

    return jsonify(res)


# 品鉴系统 执行前提：正背面图片都已输入
# 输入：data_id
@app.route("/evaluation", methods=["GET", "POST"])
def eva_stamp():
    global use_zh
    print("running eva_stamp()\n#################################################################")
    data_json = request.get_data()
    data = json.loads(data_json.decode().replace("'", "\""))
    data_id_get = data["data_id"]
    print("eva_stamp() data_id_get =", data_id_get, flush=True)

    stamp_queried = StampImage.query.filter(StampImage.data_id == data_id_get).first()
    if not stamp_queried:
        res = {"errorCode": "200013", "errorMessage": "数据库中不存在 data_id 使 data_id_get == data_id"}
        return res

    data_id = stamp_queried.data_id
    front = stamp_queried.front
    front_com = stamp_queried.front_com
    front_pos = stamp_queried.front_pos
    front_perf = stamp_queried.front_perf
    perf_dir = stamp_queried.perf_dir
    cls_name = stamp_queried.cls_name
    mat_com_b = stamp_queried.mat_com_b
    mat_com = np.frombuffer(mat_com_b, dtype=float).reshape(3, 3)
    print("eva_stamp() mat_com", mat_com, flush=True)
    back_com = stamp_queried.back_com

    print("eva_stamp() data_id", data_id, flush=True)
    print("eva_stamp() front_com, back_com", front_com, back_com, flush=True)

    if not front_com or not back_com:
        res = {"errorCode": "200001", "errorMessage": "数据库未找到用户输入图片的保存路径"}
        return jsonify(res)

    front_labelled_path, front_score_res = detectFront(front,
                                                       front_com,
                                                       front_pos,
                                                       front_perf,
                                                       perf_dir,
                                                       cls_name,
                                                       mat_com,
                                                       use_zh)
    front_labelled_name = os.path.basename(front_labelled_path)
    back_labelled_path, back_score_res = detectBack(back_com,
                                                    cls_name,
                                                    use_zh)
    back_labelled_name = os.path.basename(back_labelled_path)
    score_list = front_score_res + back_score_res

    final_score, grade, score_list_all = grade_total_score(score_list_str=score_list)

    score_str = ""
    for item in score_list:
        score_str += item + " "
    print("eva_stamp() score_str", score_str, flush=True)
    final_score = str(final_score)

    prefix = "http://" + flask_config.HOST + ":" + flask_config.PORT

    # 传url路径
    front_url_path = os.path.join(prefix, "eva_up", front_labelled_name).replace("\\", "/")
    print("eva_stamp() front_url_path", front_url_path, flush=True)
    back_url_path = os.path.join(prefix, "eva_up", back_labelled_name).replace("\\", "/")
    print("eva_stamp() back_url_path", back_url_path, flush=True)

    # 返回：错误码，错误信息，已标注正面图的web路径，已标注背面图的web路径，各品相分数，综合打分，综合评级
    return jsonify({"errorCode": "0",
                    "errorMessage": "OK",
                    "front_url_path": front_url_path,
                    "back_url_path": back_url_path,
                    "score_str": score_str,
                    "final_score": final_score,
                    "grade": grade})


if __name__ == '__main__':
    app.run(threaded=False, processes=1)
