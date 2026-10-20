# 让 `pytest tests/`（而不是 python -m pytest）也能 import 到仓库根目录的 harc.py。
# CI 里用的正是 pytest 控制台脚本，它不会自动把 rootdir 塞进 sys.path。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
