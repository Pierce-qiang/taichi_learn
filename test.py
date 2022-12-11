import numpy as np
import taichi as ti
ti.init(ti.cpu)
a = ti.field(ti.i32,shape=2)
a[0] = 20
a[1] = 40
b = ti.field(ti.i32,shape=2)
b = a
b[0] = 100
print(a[0])
