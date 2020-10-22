# Test code for publishing package with Cython

### Requirements for build:
```
numpy >= '1.13.3'
scipy >= '0.19.1'
joblib >= '0.13.1'
cython >= '0.28.5'
```
### Install:
```
python setup.py bdist_wheel
pip install dist\pack1-***.whl
```
### demo:
注意：对于pycharm用户，应该将demo.py移出Test_Cython目录，否则pycharm会优先从Test_Cython目录下导入pack1，导致找不到algo模块
将demo.py移出Test_Cython目录，pycharm就从安装目录导入pack1，这时才会有algo模块。

Note: For users who use Pycharm, the demo.py should be moved out of the Test_Cython before it is executed. 
Otherwise, Pycharm will import the pack1 from the Test_Cython, resulting in the error of ModuleNotFound.
```
import numpy as np
from pack1._algo import _YGY
N = 10
y = np.zeros(N, dtype=np.int32)
print(y)
c = 3
_YGY(y, 4)
print(y)
```

已知问题：在pack1中init导入函数似乎更为合适，但会导致编译失败

Problem: define YGY in \_\_init\_\_.py of pack1 seems more appropriate, but it will cause a compilation failure.

