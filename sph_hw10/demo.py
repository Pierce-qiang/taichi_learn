import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

ti.init(arch=ti.cuda)

# Use GPU for higher peformance if available
#ti.init(arch=ti.gpu, device_memory_GB=3, packed=True)

sprinkler_radius = 2.0
factor = 0.1
angle = ti.field(dtype=ti.f32, shape=())
sprinkler_one_side = ti.Vector.field(2, ti.f32, ())
sprinkler_another_side = ti.Vector.field(2, ti.f32, ())
offset= ti.Vector.field(2, ti.f32, ())
offset_perpendicular = ti.Vector.field(2, ti.f32, ())

# changeable parameters
velocity_factor = 5.0
sprinkler_outshape = 0.2
drop_size = [0.01,1]

@ti.kernel
def rotation():
    angle[None] -= 0.1
    angle[None] %= 360
    sprinkler_pos = ti.Vector([5.0,5.0])
    offset[None] = ti.Vector([ti.cos(angle[None]), ti.sin(angle[None])]) *sprinkler_radius
    sprinkler_one_side[None] = sprinkler_pos + offset[None]
    sprinkler_another_side[None] = sprinkler_pos -offset[None]
    offset_perpendicular[None] = ti.Vector([-offset[None][1], offset[None][0]])

def draw_sprinkler():
    one_side = sprinkler_one_side[None]
    another_side = sprinkler_another_side[None]
    one_side = ti.Vector([one_side.x,one_side.y])*factor
    another_side = ti.Vector([another_side.x,another_side.y])*factor
    gui.line(one_side,another_side, radius=5, color=0x00ff00)
    out_offset = offset_perpendicular[None]
    out_offset = ti.Vector([out_offset.x,out_offset.y]) *factor*sprinkler_outshape
    gui.line(one_side, one_side + out_offset, radius=5, color=0x00ff00)
    gui.line(another_side, another_side - out_offset, radius=5, color=0x00ff00)

counter = 0
if __name__ == "__main__":
    ps = ParticleSystem((512, 512))
    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        rotation()
        draw_sprinkler()
        corner = [sprinkler_one_side[None][0] + offset_perpendicular[None][0] * sprinkler_outshape * 2.0,
                  sprinkler_one_side[None][1] + offset_perpendicular[None][1] * sprinkler_outshape * 2.0]
        velocity = [offset_perpendicular[None][0] * velocity_factor, offset_perpendicular[None][1] * velocity_factor]
        ps.add_cube(lower_corner=corner,
                    cube_size=drop_size,
                    velocity=velocity,
                    density=10.0,
                    color=0x000000,# fake color
                    material=1)
        corner = [sprinkler_another_side[None][0] - offset_perpendicular[None][0] * sprinkler_outshape * 2.0,
                  sprinkler_another_side[None][1] - offset_perpendicular[None][1] * sprinkler_outshape * 2.0]
        velocity = [-offset_perpendicular[None][0] * velocity_factor, -offset_perpendicular[None][1] * velocity_factor]
        ps.add_cube(lower_corner=corner,
                    cube_size=drop_size,
                    velocity=velocity,
                    density=10.0,
                    color=0x000000,# fake!
                    material=1)
        for i in range(5):
            wcsph_solver.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x4169E1)
        gui.show()
        counter += 1