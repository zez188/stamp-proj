# -*- coding: UTF-8 -*-

import os


class Config(object):
    # CSRF_ENABLED = True
    # SECRET_KEY = 'you-will-never-guess'

    # 设置加密字符
    SECRET_KEY = os.urandom(16)
    DEBUG = False
    # 数据编码
    JSON_AS_ASCII = False

    @staticmethod
    def init_app(app):
        print("Config init")


class DevelopmentConfig(Config):
    DEBUG = True

    """配置参数"""
    MYSQL_DIALECT = 'mysql'  # 使用哪个数据库
    MYSQL_DIRVER = 'pymysql'  # 选择驱动
    MYSQL_NAME = 'root'  # 用户名
    MYSQL_PWD = '369963'  # 密码
    MYSQL_HOST = 'localhost'  # 主机名
    MYSQL_PORT = 3306  # 端口号
    MYSQL_DB = 'StampImage'  # 数据库名
    MYSQL_CHARSET = 'utf8mb4'  # 编码格式

    SQLALCHEMY_DATABASE_URI = f'{MYSQL_DIALECT}+{MYSQL_DIRVER}://{MYSQL_NAME}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset={MYSQL_CHARSET}'
    # SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:@localhost/StampImage'
    # 是否设置sqlalchemy自动更跟踪数据库
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 查询时会显示原始SQL语句
    SQLALCHEMY_ECHO = True
    # 禁止自动提交数据处理
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    # 设置连接池里的连接的回收时间
    SQLALCHEMY_POOL_RECYCLE = 750

    @staticmethod
    def init_app(app):
        print("DevelopmentConfig init")


class TestConfig(Config):
    TEST = True
    DEBUG = False
    """配置参数"""
    MYSQL_DIALECT = 'mysql'  # 使用哪个数据库
    MYSQL_DIRVER = 'pymysql'  # 选择驱动
    MYSQL_NAME = 'root'  # 用户名
    MYSQL_PWD = '369963'  # 密码
    MYSQL_HOST = 'localhost'  # 主机名
    MYSQL_PORT = 3306  # 端口号
    MYSQL_DB = 'StampImageTest'  # 数据库名
    MYSQL_CHARSET = 'utf8mb4'  # 编码格式

    SQLALCHEMY_DATABASE_URI = f'{MYSQL_DIALECT}+{MYSQL_DIRVER}://{MYSQL_NAME}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset={MYSQL_CHARSET}'
    # SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:@localhost/StampImage'
    # 是否设置sqlalchemy自动更跟踪数据库
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 查询时会显示原始SQL语句
    SQLALCHEMY_ECHO = True
    # 禁止自动提交数据处理
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    SQLALCHEMY_POOL_RECYCLE = 750

    @staticmethod
    def init_app(app):
        print("TestConfig init")


config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig,
    'test': TestConfig
}
