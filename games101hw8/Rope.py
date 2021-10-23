import taichi as ti

@ti.data_oriented
class Rope:
    def __init__(self, start, end, num_nodes, node_mass, ela_coeff, rest_length) -> None:
        self.start = ti.Vector(start)
        self.end = ti.Vector(end)
        self.k = ela_coeff
        self.n = num_nodes
        self.mass = node_mass
        self.rest_length = rest_length
        self.pos = ti.Vector.field(2, ti.f32, shape=self.n)
        self.lastpos = ti.Vector.field(2, ti.f32, shape=self.n)
        self.force = ti.Vector.field(2, ti.f32, shape=self.n)
        self.vel = ti.Vector.field(2, ti.f32, shape=self.n)
        #constant
        self.gravity = ti.Vector([0.0,-0.02])
        self.damping_factor = 0.005
    def display(self, gui, radius=2, color=0xffffff):
        gui.circles(self.pos.to_numpy(), radius=radius, color=color)
        for i in range(self.n-1):
            gui.line(self.pos[i], self.pos[i+1], color=0xFFFFFF, radius=1)

    @ti.kernel
    def init(self):
        for i in range(self.n):
            current_pos = self.start + i * (self.end - self.start) / (self.n - 1)
            self.pos[i] = current_pos
            self.lastpos[i] = current_pos

    @ti.kernel
    def computer_spring_force(self):
        for i in range(self.n):
            self.force[i] = ti.Vector([0.0,0.0])
        for i in range(self.n-1):
            dir = self.pos[i+1] - self.pos[i]
            force = self.k * (dir.norm() - self.rest_length) * dir/dir.norm()
            self.force[i] += force
            self.force[i+1] += -force

    @ti.kernel
    def update(self, delta_t:ti.f32, mode:ti.int8):
        for i in range(self.n):
            if i >5 :
                self.force[i] += self.mass * self.gravity
                a = self.force[i] / self.mass

                # explicit
                if mode == 0:
                    lastV = self.vel[i]
                    self.vel[i] += a * delta_t
                    self.pos[i] += lastV * delta_t

                #semi implicit
                elif mode == 1:
                    self.vel[i] += a * delta_t
                    self.pos[i] += self.vel[i] * delta_t
                # explicit Verlet
                elif mode == 2:
                    lastpos = self.pos[i]
                    self.pos[i] = self.pos[i] + (self.pos[i] - self.lastpos[i]) + 0.5 * a * delta_t * delta_t
                    self.lastpos[i] = lastpos
                # explicit Verlet with damping
                elif mode == 3:
                    lastpos = self.pos[i]
                    self.pos[i] = self.pos[i] + (1.0- self.damping_factor) * (self.pos[i] - self.lastpos[i]) + 0.5 * a * delta_t * delta_t
                    self.lastpos[i] = lastpos