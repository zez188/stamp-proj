# -*- coding: UTF-8 -*-

import multiprocessing
import os
#
# worker_class = 'gevent'
# workers = multiprocessing.cpu_count() * 2 + 1
#
# worker_connections = 1000
# timeout = 30
# max_requests = 1000
# graceful_timeout = 30
#
# loglevel = 'debug'
#
# reload = True
# debug = False
#
# bind = "%s:%s" % ("0.0.0.0", 8000)
#
# errorlog = '/tmp/error.log'
# accesslog = '/tmp/accesslog.log'


if not os.path.exists('log'):
    os.mkdir('log')

bind = '0.0.0.0:5010'                                           # gunicorn监控的接口
workers = 1                   # 进程数
threads = 1                 # 每个进程开启的线程数
proc_name = 'stamp_flask_mysql'
pidfile = './app_v1.pid'       # gunicorn进程id，kill掉该文件的id，gunicorn就停止
loglevel = 'debug'
accesslog = './log/access.log'     # access日志
errorlog = './log/error.log'    # 错误信息日志
capture_output = True
timeout = 90
reload = True
keepalive = 75              # needs to be longer than the ELB idle timeout
# worker_class = 'gevent'     # 工作模式协程
daemon = False
# 最大客户端并发数量，默认情况下这个值为1000。此设置将影响gevent和eventlet工作模式
max_requests = 20
# max_requests_jitter
max_requests_jitter = 2
