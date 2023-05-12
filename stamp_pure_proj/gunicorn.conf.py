# -*- coding: UTF-8 -*-

# import multiprocessing
import os
#
# worker_class = 'gevent'
# workers = multiprocessing.cpu_count() * 2 + 1
#
# worker_connections = 1000
# timeout = 30
# max_requests = 1000

if not os.path.exists('log'):
    os.mkdir('log')

bind = '0.0.0.0:5000'                                           # gunicorn监控的接口
workers = 1                   # 进程数
threads = 1                 # 每个进程开启的线程数
proc_name = 'stamp_flask'
pidfile = './app.pid'       # gunicorn进程id，kill掉该文件的id，gunicorn就停止
loglevel = 'debug'
accesslog = './log/access.log'     # access日志
errorlog = './log/error.log'    # 错误信息日志
capture_output = True
timeout = 90
reload = True
keepalive = 75              # needs to be longer than the ELB idle timeout
# worker_class = 'gevent'     # 工作模式协程
daemon = False

