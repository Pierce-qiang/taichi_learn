import taichi as ti
from Rope import Rope

if __name__ == "__main__":
    ti.init(arch=ti.cuda)

    # control
    paused = False
    rest_length = 0.1
    num = 20
    mode = 3
    delta_t = 0.1
    rope = Rope(start = [0.2,0.8], end = [0.8,0.8],num_nodes = num, node_mass = 1, ela_coeff = 10, rest_length = 0.02)
    rope.init()
    # GUI
    my_gui = ti.GUI("Rope", (600, 600))
    while my_gui.running:

        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == 'r':
                rope.init()
            elif e.key == 'w':
                mode = (mode+1)%4
            elif e.key == 's':
                mode = (mode-1)%4
            elif e.key == 'a':
                delta_t -= 0.01
                delta_t = max(0.001,delta_t)
            elif e.key == 'd':
                delta_t += 0.01
        if not paused:
            rope.computer_spring_force()
            rope.update(delta_t,mode)

        my_gui.text(content=f'mode 0: Simple explicit', pos=(0, 1.00), color=0xffaa77, font_size=24)
        my_gui.text(content=f'mode 1: Semi implicit', pos=(0, 0.95), color=0xffaa77, font_size=24)
        my_gui.text(content=f'mode 2: Explicit Verlet', pos=(0, 0.90), color=0xffaa77, font_size=24)
        my_gui.text(content=f'mode 3: Explicit Verlet with damping', pos=(0, 0.85), color=0xffaa77, font_size=24)
        my_gui.text(content=f'The mode is Mode {mode}', pos=(0, 0.80), color=0xffaa77, font_size=20)
        my_gui.text(content=f'delta_t is {delta_t:.3f} s', pos=(0, 0.75), color=0xffaa77, font_size=20)
        rope.display(my_gui)
        my_gui.show()