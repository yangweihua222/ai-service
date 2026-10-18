import os
import sys

# 让 ai-service 根目录进入 path，使 `from app...` 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
