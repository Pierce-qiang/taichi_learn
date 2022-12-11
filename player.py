import numpy as np
import taichi as ti


class player:
    def __init__(self, ball_choose, line_color):
        # -1：未选色；0 ：选花色球；1：选全色球
        self.ball_choose_finish = 0
        self.ball_choose = ti.field(ti.int32, shape=2)
        self.ball_choose[0] = ball_choose[0]
        self.ball_choose[1] = ball_choose[1]
        self.line_color = line_color
        self.hit_black  = ti.field(ti.int32, shape=2) #0不可打黑八，1可以
        self.hit_black[0] = 0
        self.hit_black[1] = 0

        self.target_ball = ti.field(ti.i32, shape=(2,7))

        for i in range(2):
            for j in range(7):
                self.target_ball[i,j] = 8*i+1+j


    
    def update_target(self, ball_choose):
        self.color = ball_choose
        