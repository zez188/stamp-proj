# -*- coding: UTF-8 -*-

import sys
import os

sys.path.append(os.path.abspath("yolov5_6_1"))
sys.path.append(os.path.abspath("yolov5_obb_crease"))

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

import json
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

# 定义全局变量类
class GlobalValues(object):
    data_id = ""
    front = ""
    back = ""
    front_com = ""
    back_com = ""
    front_perf = ""
    perf_dir = ""
    front_pos = ""
    front_labeled = ""
    back_labeled = ""
    mat_com = None
    cls_name = None
    use_zh = True


gv = GlobalValues()

# 标准库文件在服务器的本地路径,需要修改
# stb_path = os.path.abspath('../standardlib')
stb_path = os.path.abspath('../dataset-downloads/standardlib')
assert os.path.exists(stb_path), "标准库路径导入错误"

eva_path = "./save_anchored_img_dir"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.debug = False  # Flask内置了调试模式，可以自动重载代码并显示调试信息
app.config['JSON_AS_ASCII'] = False  # 解决flask接口中文数据编码问题

# 设置可跨域范围
CORS(app, supports_credentials=True)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


def allowed_file(filename):
    """
    判断文件类型是否允许上传
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'stamp_proj'


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
    # base64传大图片完全不行！！！
    # data_json = request.get_data()
    # resParm = json.loads(data_json)
    #
    # gv.data_id = str(uuid.uuid4())
    # print("upload() gv.data_id", gv.data_id)
    # front_name = resParm["front_name"]
    # back_name = resParm["back_name"]
    # front_base64 = resParm["front_img"]
    # back_base64 = resParm["back_img"]
    print("running upload()\n#################################################################")
    gv.data_id = str(uuid.uuid4())
    print("upload() gv.data_id", gv.data_id, flush=True)
    # 改用 form data
    front_f = request.files['front_img']
    back_f = request.files['back_img']
    # data = request.form.to_dict()
    # front_name = data["front_name"]
    # back_name = data["back_name"]

    if not front_f and not back_f:
        res = {"errorCode": "100003", "errorMessage": "正面和背面图片已损坏", "data_id": gv.data_id}
        return res
    elif not front_f:
        res = {"errorCode": "100001", "errorMessage": "正面图片已损坏", "data_id": gv.data_id}
        return res
    elif not back_f:
        res = {"errorCode": "100002", "errorMessage": "背面图片已损坏", "data_id": gv.data_id}
        return res

    front_secure_name = secure_filename(front_f.filename)
    front_ext = front_secure_name.rsplit('.')[-1]
    front_new_name = str(uuid.uuid4()) + "." + front_ext
    front_rel = os.path.join(app.config['UPLOAD_FOLDER'], front_new_name)
    gv.front = os.path.abspath(front_rel).replace("\\", "/")
    front_f.save(gv.front)
    # front_img = np.fromstring(front_b, np.uint8)
    # h,w = front_img.shape[0:2]
    # print("h, w", h, w)
    print("gv.front", gv.front, flush=True)
    # cv2.imencode('.jpg', front_img)[1].tofile(gv.front)
    pre_handle_img(gv.front, "front")

    back_secure_name = secure_filename(back_f.filename)
    back_ext = back_secure_name.rsplit('.')[-1]
    back_new_name = str(uuid.uuid4()) + "." + back_ext
    back_rel = os.path.join(app.config['UPLOAD_FOLDER'], back_new_name)
    gv.back = os.path.abspath(back_rel).replace("\\", "/")
    back_f.save(gv.back)
    # back_img = np.fromstring(back_b, np.uint8)
    print("gv.back", gv.back, flush=True)
    # cv2.imencode('.jpg', back_img)[1].tofile(gv.back)
    pre_handle_img(gv.back, "back")

    res = {"errorCode": "0", "errorMessage": "OK", "data_id": gv.data_id}
    return jsonify(res)


# 预处理
def pre_handle_img(file_abs, side):
    if side == "front":
        gv.front_com, gv.mat_com = adjust_single_img(file_abs)
        gv.front_perf, _ = adjust_perf_img(file_abs)
        gv.front_perf = gv.front_perf.replace("\\", "/")
        gv.perf_dir = cut_margin.cut_perf_adj_margin(gv.front_com,
                                                     gv.front_perf).replace("\\", "/")
        gv.front_pos = adjust_pat_img(file_abs).replace("\\", "/")
    if side == "back":
        gv.back_com, _ = adjust_single_img(file_abs)
        gv.back_com = gv.back_com.replace("\\", "/")


# 分类系统 执行前提：正面图片已输入
# 输入：data_id
@app.route("/classification", methods=["GET", "POST"])
def cls_stamp():
    print("running cls_stamp()\n#################################################################")
    data_json = request.get_data()
    print("data_json", data_json, flush=True)
    print("data_json.decode()", data_json.decode(), flush=True)
    data = json.loads(data_json.decode().replace("'", "\""))
    print("data", data, flush=True)
    data_id_get = data["data_id"]
    # data_id_get = data.get("data_id")
    print("cls_stamp() data_id_get =", data_id_get, flush=True)
    print("cls_stamp() gv.data_id", gv.data_id, flush=True)
    print("gv.front, gv.back", gv.front, gv.back, flush=True)
    if not data_id_get:
        res = {"errorCode": "200011", "errorMessage": "data_id_get = None"}
        return jsonify(res)

    if not gv.data_id:
        res = {"errorCode": "200012", "errorMessage": "gv.data_id = None"}
        return jsonify(res)

    if data_id_get != gv.data_id:
        res = {"errorCode": "200013", "errorMessage": "data_id_get != gv.data_id"}
        return jsonify(res)

    if not gv.front:
        res = {"errorCode": "200001", "errorMessage": "没有找到对应图片"}
        return jsonify(res)

    gv.cls_name, cls_prob = predict_0720.main(gv.front)

    json_cls_name_path = r"class_indices.json"
    with open(json_cls_name_path, 'r', encoding='utf-8') as fp:
        dict_cls = json.load(fp)
    cls_name_zh = dict_cls[gv.cls_name]

    cls_result = os.path.join(stb_path, cls_name_zh)
    cls_result = os.path.abspath(cls_result).replace("\\", "/")

    cls_prob_thres = 0.85
    if cls_prob > cls_prob_thres and os.path.exists(cls_result):
        if "T46.1-1" in gv.cls_name:
            cls_str = "分类结果: " + gv.cls_name + "(庚申猴票)"
        else:
            cls_str = "分类结果: " + gv.cls_name
    else:
        # cls_str = "该邮票在标准库中匹配失败,未能识别"
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
    print("running eva_stamp()\n#################################################################")
    data_json = request.get_data()
    # data = json.loads(data_json.replace("'", "\""))
    data = json.loads(data_json.decode().replace("'", "\""))
    data_id_get = data["data_id"]
    print("eva_stamp() data_id_get =", data_id_get, flush=True)
    print("eva_stamp() gv.data_id", gv.data_id, flush=True)
    print("gv.front_com, gv.back_com", gv.front_com, gv.back_com, flush=True)
    if not data_id_get:
        res = {"errorCode": "300011", "errorMessage": "data_id_get = None"}
        return jsonify(res)

    if not gv.data_id:
        res = {"errorCode": "300012", "errorMessage": "gv.data_id = None"}
        return jsonify(res)

    if data_id_get != gv.data_id:
        res = {"errorCode": "300013", "errorMessage": "data_id_get != gv.data_id"}
        return jsonify(res)

    if not gv.front_com or not gv.back_com:
        res = {"errorCode": "200001", "errorMessage": "没有找到对应图片"}
        return jsonify(res)

    front_labelled_path, front_score_res = detectFront(gv.front,
                                                       gv.front_com,
                                                       gv.front_pos,
                                                       gv.front_perf,
                                                       gv.perf_dir,
                                                       gv.cls_name,
                                                       gv.mat_com,
                                                       gv.use_zh)
    front_labelled_name = os.path.basename(front_labelled_path)
    back_labelled_path, back_score_res = detectBack(gv.back_com,
                                                    gv.cls_name,
                                                    gv.use_zh)
    back_labelled_name = os.path.basename(back_labelled_path)
    score_list = front_score_res + back_score_res

    final_score, grade, score_list_all = grade_total_score(score_list_str=score_list)

    score_str = ""
    for item in score_list:
        score_str += item + " "
    print("score_str", score_str, flush=True)
    final_score = str(final_score)

    prefix = "http://" + flask_config.HOST + ":" + flask_config.PORT

    # 传url路径
    front_url_path = os.path.join(prefix, "eva_up", front_labelled_name).replace("\\", "/")
    print("front_url_path", front_url_path, flush=True)
    back_url_path = os.path.join(prefix, "eva_up", back_labelled_name).replace("\\", "/")
    print("back_url_path", back_url_path, flush=True)

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
