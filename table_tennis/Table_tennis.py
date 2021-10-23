import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

width = 300
height = 600
res = (width,height)
tennis_radius = 10.0
hole_radius = 2*tennis_radius

#physics parameter
friction_coeff = 0.1
delta_t = 0.1

@ti.data_oriented
class Table_tennis: # all ball number = 15+1
    def __init__(self,friction_coeff, hole_radius, ball_radius, width, height):
        self.last_pos = ti.Vector.field(2, ti.f32, 16)
        self.pos = ti.Vector.field(2, ti.f32, 16)
        self.vel = ti.Vector.field(2, ti.f32, 16)
        self.ball_radius = ball_radius
        self.hole_radius = hole_radius
        self.hole_pos = ti.Vector.field(2, ti.f32, 6)
        self.roll_in = ti.field(ti.i32, shape=16) # notice 0:not in   1: roll in
        self.score = ti.field(ti.f32, shape=())
        self.friction_coeff = friction_coeff
        self.width = width
        self.height = height
    @ti.kernel
    def init(self):
        self.pos[0] = ti.Vector([0.5 * self.width,0.2 * self.height])
        center = ti.Vector([0.5 * self.width,0.5 * self.height])
        index = 1
        for z in range(1):
            for i in range(5):
                pos = ti.Vector([0.5 * self.width,0.5 * self.height])
                pos.y = center.y + i * self.ball_radius * 1.732
                pos.x = center.x - (i+2) * self.ball_radius
                for j in range(i+1):
                    pos.x += 2*self.ball_radius
                    self.pos[index] = pos
                    index += 1
        self.hole_pos[0] = ti.Vector([0.0, 0.0])
        self.hole_pos[1] = ti.Vector([0.0, 0.5])
        self.hole_pos[2] = ti.Vector([0.0, 1.0])
        self.hole_pos[3] = ti.Vector([1.0, 0.0])
        self.hole_pos[4] = ti.Vector([1.0, 0.5])
        self.hole_pos[5] = ti.Vector([1.0, 1.0])

        for i in range(6):
            self.hole_pos[i].x *= width
            self.hole_pos[i].y *= height

        for k in range(16):
            self.vel[k] = ti.Vector([0.0,0.0])
            self.last_pos[k] = self.pos[k]
            self.roll_in[k] = 0


    @ti.func
    def collision_balls(self):
        for i in range(16):
            if self.roll_in[i] == 0:
                posA = self.pos[i]
                velA = self.vel[i]
                for j in range(i+1,16):
                    if self.roll_in[i] == 0:
                        posB = self.pos[j]
                        velB = self.vel[j]
                        dir = posA - posB
                        delta_x = dir.norm()
                        if delta_x < 2*self.ball_radius:
                            # !! need to deal with ball inside
                            dir = dir/delta_x # point to A
                            normaldir = ti.Vector([dir.y, -dir.x])
                            self.vel[i] = velB.dot(dir) *dir + velA.dot(normaldir)*normaldir
                            self.vel[j] = velA.dot(dir) * dir + velB.dot(normaldir)*normaldir
                            offset = dir * (2*self.ball_radius - delta_x) *0.5
                            self.pos[i] += offset
                            self.pos[j] -= offset
    @ti.func
    def check_boundary(self,index):
        if self.pos[index].x > width - self.ball_radius or self.pos[index].x < self.ball_radius:
            self.vel[index].x *= -1
        elif  self.pos[index].y < self.ball_radius or self.pos[index].y  > height-self.ball_radius:
            self.vel[index].y *= -1
    @ti.func
    def check_roll_in(self,index) ->ti.i32:
        res = 0
        for j in range(6):  # check roll in
            dis = self.hole_pos[j] - self.pos[index]
            if dis.norm() < self.hole_radius:
                self.roll_in[index] += 1
                self.score[None] += 1
                self.vel[index] = ti.Vector([0.0, 0.0])
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
        x = max(x, 0.0)
        return ti.sqrt(x)

    @ti.func
    def update_pos(self,delta_t):
        for i in range(16):
            """print(i)  #used for test
            print(self.roll_in[i])
            print(self.vel[i])"""
            if self.roll_in[i] == 0:
                delta_x = self.pos[i] - self.last_pos[i]
                pos = self.pos[i]
                vel = self.vel[i]
                x = vel.dot(vel) - delta_x.norm() * 2 * 9.8 * self.friction_coeff
                new_vel = self.safe_sqrt(x)
                if vel.norm()>0.0001:
                    dir = vel/vel.norm()
                    self.pos[i] += new_vel * delta_t * dir
                    self.last_pos[i] = pos
                    self.vel[i] = new_vel * dir
    @ti.kernel
    def update(self,delta_t:ti.f32):
        self.update_pos(delta_t)
        self.collision_balls()
        self.collision_boundary()

    @ti.kernel
    def hit(self, velocity: ti.f32, dir_x: ti.f32, dir_y: ti.f32):
        dir = ti.Vector([dir_x, dir_y])
        dir = dir / dir.norm()
        self.vel[0] = dir * velocity
    @ti.kernel
    def check_static(self) ->ti.f32:
        res = 0.0
        for i in range(16):
            if self.roll_in[i] == 0:
                res += self.vel[i].norm()
        return res

    def display(self, gui):
        pos_np = self.pos.to_numpy()
        pos_np[:, 0] /= self.width
        pos_np[:, 1] /= self.height
        plot_balls = []
        for i in range(1,16):
            if self.roll_in[i] == 0:
                plot_balls.append(pos_np[i])
        gui.circles(np.array(plot_balls), radius=self.ball_radius, color=0xff0000)
        gui.circles(np.array([pos_np[0]]), radius=self.ball_radius, color=0xffffff)
        hole_np = self.hole_pos.to_numpy()
        hole_np[:, 0] /= self.width
        hole_np[:, 1] /= self.height
        gui.circles(hole_np, radius=self.hole_radius, color=0xffff00)
        gui.text(content=f'score = {self.score}', pos=(0, 0.90), color=0xffaa77, font_size=24)
        gui.text(content=f'velocity = {velocity_size}', pos=(0, 0.80), color=0xffaa77, font_size=24)
        gui.text(content=f'angle = {dir_angle:.2f}', pos=(0, 0.70), color=0xffaa77, font_size=24)
        gui.show()

# control
table_tennis = Table_tennis(friction_coeff,hole_radius,tennis_radius,width,height)
table_tennis.init()
# GUI
my_gui = ti.GUI("table tennis", res)
velocity_size = 100.0
dir_angle = 90.0
gain_angle = 1.0
gain_vel = 10.0



def check_win():
    res = 1
    for i in range(1, 16):
        if table_tennis.roll_in[i] == 0:
            res = 0
            break
    return res

while my_gui.running:
    while table_tennis.check_static()<0.1:
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
        pos = table_tennis.pos[0] # here is reference！！！！！
        radian = dir_angle * 2 * np.pi /360
        dir = ti.Vector([np.cos(radian), np.sin(radian)]) * velocity_size
        dir.x += pos.x
        dir.y += pos.y
        my_gui.line(ti.Vector([pos.x/width,pos.y/height]),ti.Vector([dir.x/width,dir.y/height]), color=0x00ff00)
        table_tennis.display(my_gui)
    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == 'r':
            table_tennis.init()
    table_tennis.update(delta_t)
    table_tennis.display(my_gui)
    if table_tennis.roll_in[0] == 1:
        print("You Lose")
        exit()
    if check_win() > 0.01:
        print("You Win")
        exit()