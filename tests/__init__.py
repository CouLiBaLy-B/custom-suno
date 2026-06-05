# AI Music Studio - tests package marker
import sys, os
_d = os.path.abspath(os.path.join(os.path.dirname(__file__), *['..' for _ in 'tests/__init__.py'.split('/') if _]))
if _d not in sys.path: sys.path.insert(0, _d)
