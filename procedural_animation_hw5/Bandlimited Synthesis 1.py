# reference ==> https://www.shadertoy.com/view/WtScDt#

import taichi as ti

ti.init(arch = ti.cuda)

res_x = 800
res_y = 450
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))
cos_record = ti.Vector.field(3, ti.f32)
ti.root.dense(ti.i, 16).dense(ti.jk, (res_x,res_y)).place(cos_record) # 8*2 texture

ti.Vector.field(3, ti.f32, shape=(res_x, res_y))
filter_ = True

@ti.func
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)
@ti.func
def smoothstep(edge1, edge2, v):
    assert(edge1 != edge2)
    t = (v-edge1) / float(edge2-edge1)
    t = clamp(t, 0.0, 1.0)

    return (3-2 * t) * t**2


@ti.func
def fwidth(x,k,i,j):
    ddx = cos_record[k,i+1,j] - cos_record[k,i,j]
    ddy = cos_record[k,i,j+1] - cos_record[k,i,j]
    return abs(ddx) + abs(ddy)

@ti.func
def fcos(x,k,i,j):
    w = fwidth(x,k,i,j)
    res = ti.cos(x) * ti.sin(0.5* w)/(0.5*w)  #exact
    #res = ti.cos(x) * smoothstep(6.2832, 0.0, w) # approx
    return res

@ti.func
def mcos(x,k,i,j):
    res = ti.cos(x)
    if filter_:
        res = fcos(x,k,i,j)
    return res

@ti.func
def getcolor(t, k, i,j):
    col = ti.Vector([0.6, 0.5, 0.4])
    col += 0.14 * mcos(cos_record[k,i,j],k+0,i,j)
    col += 0.13 * mcos(cos_record[k+1,i,j],k+1,i,j)
    col += 0.12 * mcos(cos_record[k+2,i,j],k+2,i,j)
    col += 0.11 * mcos(cos_record[k+3,i,j],k+3,i,j)
    col += 0.10 * mcos(cos_record[k+4,i,j],k+4,i,j)
    col += 0.09 * mcos(cos_record[k+5,i,j],k+5,i,j)
    col += 0.08 * mcos(cos_record[k+6,i,j],k+6,i,j)
    col += 0.07 * mcos(cos_record[k+7,i,j],k+7,i,j)
    return col
@ti.func
def init_texture(p,i,j):
    t = p.x
    offset = 0
    for s in range(2):
        cos1 = 6.2832 * t * 1.0 + ti.Vector([0.0, 0.5, 0.6])
        cos2 = 6.2832 * t * 3.1 + ti.Vector([0.5, 0.6, 1.0])
        cos3 = 6.2832 * t * 5.1 + ti.Vector([0.1, 0.7, 1.1])
        cos4 = 6.2832 * t * 9.1 + ti.Vector([0.1, 0.5, 1.2])
        cos5 = 6.2832 * t * 17.1 + ti.Vector([0.0, 0.3, 0.9])
        cos6 = 6.2832 * t * 31.1 + ti.Vector([0.1, 0.5, 1.3])
        cos7 = 6.2832 * t * 65.1 + ti.Vector([0.1, 0.5, 1.3])
        cos8 = 6.2832 * t * 131.1 + ti.Vector([0.3, 0.2, 0.8])
        cos_record[offset,i,j] = cos1
        cos_record[offset+1, i, j] = cos2
        cos_record[offset+2, i, j] = cos3
        cos_record[offset+3, i, j] = cos4
        cos_record[offset+4, i, j] = cos5
        cos_record[offset+5, i, j] = cos6
        cos_record[offset+6, i, j] = cos7
        cos_record[offset+7, i, j] = cos8
        t = p.y
        offset = 8



@ti.kernel
def render(time:ti.f32):
    for i, j in pixels:
        q = ti.Vector([2 * i - res_x, 2 * j - res_y]) / res_y
        p = 2.0 * q / q.dot(q)
        p += 0.05 * time
        init_texture(p,i,j)
    for i,j in pixels:
        q = ti.Vector([2*i-res_x, 2*j- res_y]) / res_y

        p = 2.0 * q / q.dot(q)

        p += 0.05*time

        col = min(getcolor(p.x,0,i,j), getcolor(p.y,8,i,j))

        col *= 1.5 - 0.2*q.norm()

        pixels[i,j] = col
gui = ti.GUI("Canvas", res=(res_x, res_y))
for i in range(100000):
    t = i * 0.03
    render(t)
    gui.set_image(pixels)
    gui.show()