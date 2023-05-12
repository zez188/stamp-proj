# -*- coding: UTF-8 -*-

from flask_cors import CORS
from app import create_app, db
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand       # 载入migrate扩展
from config import config
# 可选default和DevelopmentConfig
# app = create_app('default')
config_class = config['development']
app = create_app(config_class)
CORS(app, supports_credentials=True)
migrate = Migrate(app, db)                              # 注册migrate到flask
manage = Manager(app)

manage.add_command('db', MigrateCommand)                # 在终端环境下添加一个db命令

if __name__ == '__main__':
    manage.run()

    # python manage_v1.py db init
    # python manage_v1.py runserver
