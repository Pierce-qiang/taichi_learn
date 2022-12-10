import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)
from ball import *
from table import *


@ti.data_oriented
class Table_tennis: # all ball number = 15+1
    def __init__(self,friction_coeff, hole_radius, ball_radius, width, height):

        self.ball = ball(ball_radius)
        self.table = table(hole_radius,width, height)

        self.roll_in = ti.field(ti.i32, shape=16) # notice 0:not in   1: roll in
        self.score = ti.field(ti.f32, shape=())
        self.friction_coeff = friction_coeff

    @ti.kernel
    def init(self):
        self.score[None] = 0

        self.ball.init(self.table.width, self.table.height)
        self.table.init()

        for k in range(16):
            self.ball.vel[k] = ti.Vector([0.0,0.0])
            self.ball.last_pos[k] = self.ball.pos[k]
            self.roll_in[k] = 0


    @ti.func
    def collision_balls(self):
        for i in range(16):
            if self.roll_in[i] == 0:
                posA = self.ball.pos[i]
                velA = self.ball.vel[i]
                for j in range(i+1,16):
                    if self.roll_in[i] == 0:
                        posB = self.ball.pos[j]
                        velB = self.ball.vel[j]
                        dir = posA - posB
                        delta_x = dir.norm()
                        if delta_x < 2*self.ball.ball_radius:
                            # !! need to deal with ball inside
                            dir = dir/delta_x # point to A
                            normaldir = ti.Vector([dir.y, -dir.x])
                            self.ball.vel[i] = velB.dot(dir) *dir + velA.dot(normaldir)*normaldir
                            self.ball.vel[j] = velA.dot(dir) * dir + velB.dot(normaldir)*normaldir
                            offset = dir * (2*self.ball.ball_radius - delta_x) *0.5
                            self.ball.pos[i] += offset
                            self.ball.pos[j] -= offset


    @ti.func
    def check_boundary(self,index):
        if self.ball.pos[index].x > self.table.width - self.ball.ball_radius or self.ball.pos[index].x < self.ball.ball_radius:
            self.ball.vel[index].x *= -1
        elif  self.ball.pos[index].y < self.ball.ball_radius or self.ball.pos[index].y  > self.table.height-self.ball.ball_radius:
            self.ball.vel[index].y *= -1


    @ti.func
    def check_roll_in(self,index) ->ti.i32:
        res = 0
        for j in range(6):  # check roll in
            dis = self.table.hole_pos[j] - self.ball.pos[index]
            if dis.norm() < self.table.hole_radius:
                self.roll_in[index] += 1
                self.score[None] += 1
                self.ball.vel[index] = ti.Vector([0.0, 0.0])
                res = 1
                break
        return res


    @ti.func
    def collision_boundary(self):
        for i in range(16):
            if self.roll_in[i] == 0:
                self.check_roll_in(i)
                self.check_boundary(i)


    @ti.func
    def safe_sqrt(self, x) ->ti.f32:
        x = ti.max(x, 0.0)
        return ti.sqrt(x)

    @ti.func
    def update_pos(self,delta_t):
        for i in range(16):
            """print(i)  #used for test
            print(self.roll_in[i])
            print(self.ball.vel[i])"""
            if self.roll_in[i] == 0:
                delta_x = self.ball.pos[i] - self.ball.last_pos[i]
                pos = self.ball.pos[i]
                vel = self.ball.vel[i]
                x = vel.dot(vel) - delta_x.norm() * 2 * 9.8 * self.friction_coeff
                new_vel = self.safe_sqrt(x)
                if vel.norm()>0.0001:
                    dir = vel/vel.norm()
                    self.ball.pos[i] += new_vel * delta_t * dir
                    self.ball.last_pos[i] = pos
                    self.ball.vel[i] = new_vel * dir


    @ti.kernel
    def update(self,delta_t:ti.f32):
        self.update_pos(delta_t)
        self.collision_balls()
        self.collision_boundary()

    @ti.kernel
    def hit(self, velocity: ti.f32, dir_x: ti.f32, dir_y: ti.f32):
        dir = ti.Vector([dir_x, dir_y])
        dir = dir / dir.norm()
        self.ball.vel[0] = dir * velocity


    @ti.kernel
    def check_static(self) ->ti.f32:
        res = 0.0
        for i in range(16):
            if self.roll_in[i] == 0:
                res += self.ball.vel[i].norm()
        return res

    def display(self, gui, velocity_size, dir_angle):
        pos_np = self.ball.pos.to_numpy()
        pos_np[:, 0] /= self.table.width
        pos_np[:, 1] /= self.table.height
        plot_balls = []
        for i in range(1,16):
            if self.roll_in[i] == 0:
                plot_balls.append(pos_np[i])
        gui.circles(np.array(plot_balls), radius=self.ball.ball_radius, color=0xff0000)
        gui.circles(np.array([pos_np[0]]), radius=self.ball.ball_radius, color=0xffffff)
        hole_np = self.table.hole_pos.to_numpy()
        hole_np[:, 0] /= self.table.width
        hole_np[:, 1] /= self.table.height
        gui.circles(hole_np, radius=self.table.hole_radius, color=0xffff00)
        gui.text(content=f'score = {self.score}', pos=(0, 0.90), color=0xffaa77, font_size=24)
        gui.text(content=f'velocity = {velocity_size}', pos=(0, 0.80), color=0xffaa77, font_size=24)
        gui.text(content=f'angle = {dir_angle:.2f}', pos=(0, 0.70), color=0xffaa77, font_size=24)
        gui.show()
