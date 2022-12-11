import numpy as np
import taichi as ti

#1到15号共15颗目标球，1到7号球为全色球，8号为黑色球，9到15号为双色球（又称花色球）。
#0号为白色球

class ball:
    def __init__(self, ball_radius):
        self.ball_radius = ball_radius
        self.last_pos = ti.Vector.field(2, ti.f32, 16)
        self.pos = ti.Vector.field(2, ti.f32, 16)
        self.vel = ti.Vector.field(2, ti.f32, 16)
        self.vel = ti.Vector.field(2, ti.f32, 16)

        self.color = [0xffffff,0xFF1493,0xFF1493,0xFF1493,0xFF1493,0xFF1493,0xFF1493,0xFF1493,
        0x000000,0x8B4513,0x8B4513,0x8B4513,0x8B4513,0x8B4513,0x8B4513,0x8B4513]


    def init(self,table_width,table_height):
        self.pos[0] = ti.Vector([0.2 * table_width,0.5 * table_height])
        center = ti.Vector([0.7 * table_width,0.5 * table_height])
        index = 1
        for z in range(1):
            for i in range(5):
                pos = ti.Vector([0.5 * table_width,0.5 * table_height])
                pos.x = center.x + i * self.ball_radius * 1.732
                pos.y = center.y - (i+2) * self.ball_radius
                for j in range(i+1):
                    pos.y += 2*self.ball_radius
                    self.pos[index] = pos
                    index += 1

        