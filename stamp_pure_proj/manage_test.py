# -*- coding: UTF-8 -*-

import requests
from threading import Thread
import time


def _req():
    return requests.get('http://localhost:5000/').text


pre_time = time.time()

threads = [Thread(target=_req) for _ in range(3)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print(time.time() - pre_time, 's')
