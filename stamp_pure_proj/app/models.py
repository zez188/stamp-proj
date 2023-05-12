# -*- coding: UTF-8 -*-

from . import db


class StampImage(db.Model):
    # 定义表名
    __tablename__ = 'stampImage'

    # 定义字段
    data_id = db.Column(db.String(64), primary_key=True)
    front = db.Column(db.String(128))
    back = db.Column(db.String(128))
    front_com = db.Column(db.String(128))
    back_com = db.Column(db.String(128))
    front_perf = db.Column(db.String(128))
    perf_dir = db.Column(db.String(128))
    front_pos = db.Column(db.String(128))
    # front_labelled = db.Column(db.String(128))
    # back_labelled = db.Column(db.String(128))
    mat_com_b = db.Column(db.PickleType)
    cls_name = db.Column(db.String(64))

    # repr()方法显示一个可读字符串，不是完全必要，可用于调试和测试。
    def __repr__(self):
        return '<stamp_image data_id {}> '.format(self.data_id)

    @classmethod
    def create_one(cls,
                   data_id,
                   front,
                   front_com,
                   front_perf,
                   front_pos,
                   perf_dir,
                   mat_com_b,
                   back,
                   back_com
                   ):
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

        return tmp_stamp

    @classmethod
    def get_by_id(cls, data_id):
        return StampImage.query.filter_by(data_id=data_id).first()

    # @classmethod
    # def get_front_back(cls, data_id):
    #     tmp_stamp = StampImage.get_by_id(data_id=data_id)
    #     if tmp_stamp is None:
    #         return False, False
    #
    #     return tmp_stamp.front, tmp_stamp.back

    @classmethod
    def update_one_cls(cls, data_id, cls_name):
        tmp_stamp = StampImage.get_by_id(data_id=data_id)

        if tmp_stamp is None:
            return False

        tmp_stamp.cls_name = cls_name

        db.session.add(tmp_stamp)
        db.session.commit()

        return tmp_stamp

    # @classmethod
    # def update_one_eva(cls, data_id, front_labelled, back_labelled):
    #     tmp_stamp = StampImage.get_by_id(data_id=data_id)
    #
    #     if tmp_stamp is None:
    #         return False
    #
    #     tmp_stamp.front_labelled = front_labelled
    #     tmp_stamp.back_labelled = back_labelled
    #
    #     db.session.add(tmp_stamp)
    #     db.session.commit()
    #
    #     return tmp_stamp

    @classmethod
    def delete_one(cls, data_id):
        tmp_stamp = StampImage.get_by_id(data_id=data_id)

        if tmp_stamp is None:
            return False

        db.session.delete(tmp_stamp)
        db.session.commit()

        return True
