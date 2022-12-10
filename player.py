import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

@ti.data_oriented
class player:
    def __init__(self, ball_choose):
        self.ball = ball_choose  # 0：未选色；1 ：选花色球；2：选全色球
