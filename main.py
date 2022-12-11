import numpy as np
import taichi as ti
from Table_tennis import *

ti.init(arch=ti.cpu)


#定义台球游戏中尺寸信息
tb_origin_width = 2830
tb_origin_height = 1550
reduce_scale  = 4
tennis_origin_radius = 57/2
hole_origin_redius = 150/2
tennis_radius = tennis_origin_radius/reduce_scale
hole_radius = hole_origin_redius/reduce_scale

width = tb_origin_width//reduce_scale
height = tb_origin_height//reduce_scale
res = (width,height)


#physics parameter
friction_coeff = 0.1
delta_t = 0.1

# control
table_tennis = Table_tennis(friction_coeff,hole_radius,tennis_radius,width,height)
table_tennis.init()
# GUI

my_gui = ti.GUI("table tennis", res,table_tennis.background_color)
velocity_size = 100.0
dir_angle = 0
gain_angle = 1.0
gain_vel = 10.0


def check_win():
    res = 1
    for i in range(1, 16):
        if table_tennis.roll_in[i] == 0:
            res = 0
            break
    return res

first_static = 0
while my_gui.running:
    while table_tennis.check_static()<0.1:
        if first_static == 1:
            table_tennis.hit_finish()
            table_tennis.first_collision=0
            table_tennis.first_hit = -1
            first_static = 0
            print('stop')
            if table_tennis.game_state == 0:
                print("Player 1 win")
                exit()
            if table_tennis.game_state == 1:
                print("Player 2 win")
                exit()

        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == 'r':
                table_tennis.init()
            elif e.key == 'a':
                dir_angle += 1*gain_angle
                dir_angle %= 360
            elif e.key == 'd':
                dir_angle -= 1*gain_angle
                dir_angle %= 360
            elif e.key == 'w':
                velocity_size += 1.0 * gain_vel
                velocity_size = min(velocity_size,200)
            elif e.key == 's':
                velocity_size -= 1.0 * gain_vel
                velocity_size = max(0.0, velocity_size)
            elif e.key == '1':
                gain_angle += 10.0
            elif e.key == '2':
                gain_angle -= 10.0
            elif e.key == '3':
                gain_vel += 10.0
            elif e.key == '4':
                gain_vel -= 10.0
            elif e.key == 'z':
                radian = dir_angle * 2 * np.pi /360
                table_tennis.hit(velocity_size, np.cos(radian), np.sin(radian))
                table_tennis.in_hit = 1
                table_tennis.first_collision = 0
                table_tennis.first_hit = 0
                first_static = 1
            

        
        #做出击球线
        pos = table_tennis.ball.pos[0]  
        radian = dir_angle * 2 * np.pi /360
        dir = ti.Vector([np.cos(radian), np.sin(radian)]) * velocity_size
        dir.x += pos.x
        dir.y += pos.y
        my_gui.line(ti.Vector([pos.x/width,pos.y/height]),ti.Vector([dir.x/width,dir.y/height]), color=table_tennis.line_color[table_tennis.now_player[0]])
        table_tennis.display(my_gui, velocity_size, dir_angle)




    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == 'r':
            table_tennis.init()
    table_tennis.update(delta_t)
    table_tennis.collision_white_balls()
    table_tennis.display(my_gui, velocity_size, dir_angle)

