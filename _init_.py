#coding=utf-8
#__init__.py 文件的作用是将文件夹变为一个Python模块,Python 中的每个模块的包中，都有__init__.py 文件.
# 通常__init__.py 文件为空，但是我们还可以为它增加其他的功能。我们在导入一个包时，实际上是导入了它的__init__.py文件。
# 这样我们可以在__init__.py文件中批量导入我们所需要的模块，而不再需要一个一个的导入。
# __init__.py主要控制包的导入行为
# package
# __init__.py
import re
import urllib
import sys
import os
# a.py
import package
print(package.re, package.urllib, package.sys, package.os)