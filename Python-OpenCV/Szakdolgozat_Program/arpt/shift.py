import numpy as np

class shift():
    def __init__(self, pos_x, pos_y, width, height):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def calc_shift(self, grid, cap):
        x = np.uint8(np.floor(self.pos_x / grid.grid_step))
        y = np.uint8(np.floor(self.pos_y / grid.grid_step))
        w = np.uint8(np.floor(self.width / grid.grid_step))
        h = np.uint8(np.floor(self.height / grid.grid_step))

        localVectorSum = np.array([ grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                                    grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                                    dtype=np.float32).sum(axis=1)

        localDirectionVector = np.subtract(localVectorSum[1], localVectorSum[0])

        vectorCount = len(localVectorSum)

        self.velocity_x += localDirectionVector[0] / vectorCount*0.5
        self.velocity_y += localDirectionVector[1] / vectorCount*0.5

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