import numpy as np
from arpt.vector import vector

class shift():
    def __init__(self, pos_x, pos_y, width, height):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def calc_shift(self, grid, cap):
        x,y,w,h = np.uint8(np.floor(np.divide((self.pos_x,
                                                self.pos_y,
                                                self.width,
                                                self.height), grid.grid_step)))


        local_vector_sum = vector(np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                                            grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                                            dtype=np.float32).sum(axis=1))

        local_direction_vector = local_vector_sum.dir_vector()

        vector_count = len(local_vector_sum.vector)

        self.velocity_x += local_direction_vector.vector[0] / vector_count*0.5
        self.velocity_y += local_direction_vector.vector[1] / vector_count*0.5

        self.pos_x += self.velocity_x
        self.pos_y += self.velocity_y

        self.velocity_x *= 0.8
        self.velocity_y *= 0.8

        if self.pos_x + self.width >= cap.width:
            self.pos_x = cap.width - self.width

        if self.pos_y + self.height >= cap.height:
            self.pos_y = cap.height - self.height

        if self.pos_x < 0:
            self.pos_x = 0

        if self.pos_y < 0:
            self.pos_y = 0