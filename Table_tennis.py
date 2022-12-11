import numpy as np
import taichi as ti
from ball import *
from table import *
from player import *


@ti.data_oriented
class Table_tennis: # all ball number = 15+1
    def __init__(self,friction_coeff, hole_radius, ball_radius, width, height):

        self.ball = ball(ball_radius)
        self.table = table(hole_radius,width, height)

        self.roll_in = ti.field(ti.i32, shape=16) # notice 0:not in   1: roll in
        self.last_rollin = ti.field(ti.i32, shape = 16)

        self.score = ti.field(ti.f32, shape=())
        self.friction_coeff = friction_coeff
        self.background_color = 0x3CB371
        self.hole_color = 0x000000

        self.line_color = ti.field(ti.i32, shape=2)
        self.line_color[0] = 0xB22222
        self.line_color[1] = 0xFFA500

        self.ball_choose = ti.field(ti.i32, shape=1)
        self.ball_choose[0] = 0 #0未选色，1已经选色
        self.now_player = ti.field(ti.i32, shape=1)
        self.now_player[0] = 0 #选择0或者1

        self.player = player([-1,-1], self.line_color )
        
        # self.in_hit = 0 #在一次击球过程中
        self.first_collision = 0#白球是否发生了第一次碰撞

        # self.first_hit = ti.field(ti.i32, shape=1)
        # self.first_hit[0] = 0 #第一个碰到的球
        self.first_hit = -1
        self.game_state = ti.field(ti.i32, shape=1)
        self.game_state[0] = -1 # -1: not end;0:player 1 win; 1:player 2 win
        



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


    def collision_white_balls(self):
        #还没使用
        if self.first_collision==0:

            posA = self.ball.pos[0]
            for j in range(1,16):
                if self.roll_in[0] == 0:
                    posB = self.ball.pos[j]
                    dir = posA - posB
                    delta_x = dir.norm()
                    if delta_x < 2*self.ball.ball_radius:
                        if self.first_collision == 0: #检测第一个碰到的球
                            self.first_hit = j
                            self.first_collision = 1


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


    def hit_finish(self):
        #结束一次击球，需要判断击球是否犯规
        print('first hit ball is ', self.first_hit)
        hit_result = self.change_player()
        if hit_result == 2:
            self.game_state[0] = 1-self.now_player[0]
        elif hit_result == 3:
            self.game_state[0] = self.now_player[0]
        elif hit_result == 1:
            self.now_player[0] = 1-self.now_player[0]
        else:
            pass

        print('now player is ', self.now_player[0])
        for i in range(1,16):
            self.last_rollin[i] = self.roll_in[i]

        
    

    def change_player(self) -> ti.i16:
        #先进行是否选了花色判断
        # 返回0，继续击球，返回1，交换击球，返回2，直接结束游戏（比赛输了；返回3，进黑八，比赛赢了
        change_id = 0
        wrong_hit_flag = 1
        if self.player.ball_choose_finish == 1:
            if self.player.ball_choose[self.now_player[0]] == 0:#打花色球1-7
                if self.first_hit>=1 and self.first_hit<=7:
                    wrong_hit_flag=0
            if self.player.ball_choose[self.now_player[0]] == 1:#打全色球9-15
                if self.first_hit>=9 and self.first_hit<=15:
                    wrong_hit_flag=0
                
            if self.roll_in[8] == 1:#进黑八直接结束
                if self.player.hit_black[self.now_player[0]]==1:
                    change_id =3
                    print("犯规：打进黑八，游戏结束，失败")
                else:
                    change_id = 2
                    print("游戏结束")
            elif self.roll_in[0] == 1: #进了白球
                self.free_ball()
                self.roll_in[0] = 0
                change_id =1 
                print("犯规：白球落袋，自由球")
            elif wrong_hit_flag: #先碰到了别人球
                self.free_ball()
                change_id = 1
                print("犯规：击打对方球，自由球")
            else: #正常击球
                flag = 0
                for i in range(0,16):
                    if self.last_rollin[i] != self.roll_in[i]:
                        if i >= self.player.target_ball[self.player.ball_choose[self.now_player[0]],0] and i <= self.player.target_ball[self.player.ball_choose[self.now_player[0]],6]:

                            change_id = 0
                            flag = 1
                if flag == 0:
                    change_id = 1
                    print("正常击球，交换球权")
                else:
                    print("进入目标球，继续击球")
        else: #没选花色
            if self.roll_in[8] == 1:#进黑八直接结束
                change_id = 2
            elif self.roll_in[0] == 1: #进了白球
                self.free_ball()
                self.roll_in[0] = 0
                change_id =1 
            elif self.first_hit==8: #碰了黑八
                self.free_ball()
                change_id = 1
            else: #正常击球
                #首先击中的球进了，那么就选色成功，没进就交换
                if self.first_hit>=1 and self.first_hit<=7:
                    flag = 0
                    for i in range(7):
                        if self.last_rollin[i+1] != self.roll_in[i+1]:
                            flag =1
                    
                    if flag ==1:#选色成功
                        self.player.ball_choose[self.now_player[0]] = 0
                        self.player.ball_choose[1-self.now_player[0]] = 1
                        self.player.ball_choose_finish = 1
                        change_id = 0
                        print("选色成功，玩家", self.now_player[0],"击打花色球")
                    else:
                        change_id = 1
                elif self.first_hit>=9 and self.first_hit<=15:
                    flag = 0
                    for i in range(7):
                        if self.last_rollin[i+9] != self.roll_in[i+9]:
                            flag =1
                    
                    if flag ==1:#选色成功
                        self.player.ball_choose[self.now_player[0]] = 1
                        self.player.ball_choose[1-self.now_player[0]] = 0
                        self.player.ball_choose_finish = 1
                        change_id = 0
                        print("选色成功，玩家", 1-self.now_player[0],"击打花色球")
                    else:
                        change_id = 1
                else:#没达到球
                    self.free_ball()
                    change_id = 1
                    print('first hit = ',self.first_hit)
                    print('没打到球')

        # print('change id = ', change_id)
        return change_id

    
    #这里需要可以手动选择位置

    def free_ball(self):#自由球
        self.ball.pos[0] = ti.Vector([0.2 * self.table.width,0.5 * self.table.height])


    

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


    def hit(self, velocity: ti.f32, dir_x: ti.f32, dir_y: ti.f32):
        dir = ti.Vector([dir_x, dir_y])
        dir = dir / dir.norm()
        self.ball.vel[0] = dir * velocity



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

        for i in range(0,16):
            if self.roll_in[i] == 0:
                gui.circle(pos_np[i], radius=self.ball.ball_radius, color=self.ball.color[i])

        hole_np = self.table.hole_pos.to_numpy()
        hole_np[:, 0] /= self.table.width
        hole_np[:, 1] /= self.table.height
        gui.circles(hole_np, radius=self.table.hole_radius, color=self.hole_color)
        gui.text(content=f'score = {self.score}', pos=(0, 0.90), color=0xffaa77, font_size=24)
        gui.text(content=f'velocity = {velocity_size}', pos=(0, 0.80), color=0xffaa77, font_size=24)
        gui.text(content=f'angle = {dir_angle:.2f}', pos=(0, 0.70), color=0xffaa77, font_size=24)
        gui.show()
