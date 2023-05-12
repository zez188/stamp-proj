# -*- coding: UTF-8 -*-

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy()

from .views import stamp
from .models import StampImage


def create_app(config_obj):
    """ 使用工厂函数初始化程序实例"""

    app.config.from_object(config_obj)

    app.app_context().push()

    app.register_blueprint(stamp)

    with app.app_context():
        db.init_app(app)
        db.create_all()

    return app
