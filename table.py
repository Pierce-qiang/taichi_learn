import numpy as np
import taichi as ti


class table:
    def __init__(self, hole_radius, width, height):
        self.width = width
        self.height = height
        self.hole_radius = hole_radius
        self.hole_pos = ti.Vector.field(2, ti.f32, 6)


    def init(self):
        self.hole_pos[0] = ti.Vector([0.0, 0.0])
        self.hole_pos[1] = ti.Vector([0.5, 0.0])
        self.hole_pos[2] = ti.Vector([1.0, 0.0])
        self.hole_pos[3] = ti.Vector([0.0, 1.0])
        self.hole_pos[4] = ti.Vector([0.5, 1.0])
        self.hole_pos[5] = ti.Vector([1.0, 1.0])

        for i in range(6):
            self.hole_pos[i].x *= self.width
            self.hole_pos[i].y *= self.height