import cv2
import numpy as np


class View(object):
    """
    Class for showing the values and results of other objects
    """

    def show_canvas(self, win, canvas):
        """
        Show canvas.
        """
        self.show_image(win, canvas._canvas)

    def show_image(self, win, image):
        """
        Show image.
        """
        cv2.imshow(win._name, image)

    def show_vector_field(self, grid, win, canvas):
        """
        Show the vector field.
        """
        canvas.fill(255)
        for k in range(len(grid.new_points)):
            current_vector = np.subtract(grid.new_points[k], grid._old_points[k])
            if abs(current_vector[0]) >= 2 or abs(current_vector[1]) >= 2:

                cv2.arrowedLine(    canvas._canvas,
                                    tuple(grid._old_points[k]),
                                    tuple(grid.new_points[k]),
                                    (0,0,0),
                                    2)
        self.show_canvas(win, canvas)


    def show_heat_map(self, win, heat_map):
        """
        Show the heatmap.
        """
        resized_heat_map = cv2.resize(heat_map.map, dsize=(600, 320), interpolation=cv2.INTER_AREA)

        for i in range(len(heat_map.bounding_rects)):
            (x,y,w,h) = heat_map.bounding_rects[i]
            cv2.rectangle(resized_heat_map, (x*40, y*40), (x*40+w*40, y*40+h*40), (255, 255, 255), 2)
            cv2.arrowedLine(    resized_heat_map,
                                (int(x*40 + (w*40) / 2),
                                int(y*40 + (h*40) / 2)),
                                (int(heat_map.motion_points_direction[i][0] * 100 + x*40 + (w*40) / 2),
                                int(heat_map.motion_points_direction[i][1] * 100 + y*40 + (h*40) / 2)),
                                (0, 255, 255),
                                2)
        self.show_image(win,resized_heat_map)


    def show_global_vector_results(self, grid, win, canvas):
        """
        Show the global vector results.
        """
        canvas.fill(255)

        cv2.putText(    canvas._canvas,
                        'Global Resultant Vector',
                        (250, 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)

        cv2.putText(    canvas._canvas,
                        'AVG Vector Lenght',
                        (0, 15),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)

        cv2.line(canvas._canvas, (15, 285), (470, 285), (0, 180, 0), 1)
        cv2.line(canvas._canvas, (15, 285), (15, 20), (0, 180, 0), 1)

        cv2.putText(    canvas._canvas,
                        'Direction',
                        (560, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)

        cv2.line(canvas._canvas, (600, 250), (600, 50), (0, 180, 0), 1)
        cv2.line(canvas._canvas, (500, 150), (700, 150), (0, 180, 0), 1)

        avg_leghts = np.int32(np.add(np.multiply(grid.avg_vector_lenghts, -20), 285))
        
        dots = [0, 5, 10]

        for dot in dots:
            position = np.int32(np.add(np.multiply(dot, -20), 285))
            cv2.circle(canvas._canvas, (15, position), 1, (0, 180, 0), 3)

            cv2.putText(    canvas._canvas,
                            str(dot),
                            (0,position),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA)
    
        step = 15
        i = 1
        while i < len(avg_leghts):
            cv2.line(canvas._canvas, (step*i, avg_leghts[i-1]), (step*i+step, avg_leghts[i]), (0, 0, 255), 2)
            i += 1

        cv2.line(canvas._canvas, (15, avg_leghts[-1]), (i*step, avg_leghts[-1]), (0, 120, 0), 1)

        cv2.arrowedLine(    canvas._canvas,
                            (0+600, 0+150),
                            (int(grid.global_direction_vector.vector[0]/8) + 600,
                            int(grid.global_direction_vector.vector[1]/8) + 150),
                            (0, 0, 0),
                            2)

        self.show_canvas(win, canvas)

    def show_shift(self, shift, video):
        """
        Show the shift vectors.
        """
        cv2.rectangle(  video.frame,
                        (int(shift.pos_x), int(shift.pos_y)),
                        (int(shift.pos_x) + int(shift.width), int(shift.pos_y) + shift.height),
                        (0, 255, 0),
                        3)

