import cv2
import numpy as np
import arpt.vector as v


class View(object):
    """
    Class for showing the values and results of other objects
    """

    def show_canvas(self, win, canvas):
        """
        Show canvas.
        """
        self.show_image(win, canvas.canvas)

    def show_image(self, win, image):
        """
        Show image.
        """
        cv2.imshow(win.name, image)

    def show_vector_field(self, grid, swirl, win, canvas, eps=2):
        """
        Show the vector field.
        """
        canvas.fill(255)
        for k in range(len(grid.new_points)):
            current_vector = np.subtract(grid.new_points[k],
                                         grid.old_points[k])

            if v.get_vector_length(current_vector) >= eps:
                cv2.arrowedLine(canvas.canvas,
                                tuple(grid.old_points[k]),
                                tuple(grid.new_points[k]),
                                0,
                                2)
        # NOTE: The intersection points here is for testing only!
        if swirl.points.any():
            for point in swirl._points:
                cv2.circle(canvas.canvas, tuple(point), 10, (0, 0, 255), 5)

        self.show_canvas(win, canvas)

    def show_heat_map(self, win, heat_map):
        """
        Show the heatmap.
        """
        magnification = 20
        width, height, _ = heat_map.map.shape
        resized_heat_map = cv2.resize(heat_map.map,
                                      dsize=(height*magnification,
                                             width*magnification),
                                      interpolation=cv2.INTER_AREA)

        for i in range(len(heat_map.bounding_rects)):
            (x, y, w, h) = heat_map.bounding_rects[i]
            cv2.rectangle(resized_heat_map,
                          (x*magnification, y*magnification),
                          (x*magnification+w*magnification,
                              y*magnification+h*magnification),
                          (255, 255, 255),
                          1)
            cv2.arrowedLine(resized_heat_map,
                            (int(x*magnification + (w*magnification) / 2),
                                int(y*magnification + (h*magnification) / 2)),
                            (int(heat_map.motion_points_direction[i][0] *
                                 magnification*2
                                 + x*magnification
                                 + (w*magnification) / 2),
                                int(heat_map.motion_points_direction[i][1] *
                                    magnification*2
                                    + y*magnification
                                    + (h*magnification) / 2)),
                            (0, 255, 255),
                            1)
        formatted_diff_dir_percentage = \
            float("{0:.2f}".format(heat_map.different_direction))

        if formatted_diff_dir_percentage == 0:
            formatted_diff_dir_percentage = "00.00"
        else:
            if formatted_diff_dir_percentage < 10:
                formatted_diff_dir_percentage = \
                    "0" + str(formatted_diff_dir_percentage)

        cv2.putText(resized_heat_map,
                    str(formatted_diff_dir_percentage),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        self.show_image(win, resized_heat_map)

    def show_global_vector_results(self, grid, win, canvas):
        """
        Show the global vector results.
        """
        canvas.fill(255)

        cv2.putText(canvas.canvas,
                    'Global Resultant Vector',
                    (250, 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(canvas.canvas,
                    'AVG Vector length',
                    (0, 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.line(canvas.canvas, (15, 285), (470, 285), (0, 180, 0), 1)
        cv2.line(canvas.canvas, (15, 285), (15, 20), (0, 180, 0), 1)

        cv2.putText(canvas.canvas,
                    'Direction',
                    (560, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.line(canvas.canvas, (600, 250), (600, 50), (0, 180, 0), 1)
        cv2.line(canvas.canvas, (500, 150), (700, 150), (0, 180, 0), 1)

        avg_leghts = np.int32(np.add(np.multiply(grid.avg_vector_lengths,
                                                 -20),
                                     285))

        dots = [0, 5, 10]

        for dot in dots:
            position = np.int32(np.add(np.multiply(dot, -20), 285))
            cv2.circle(canvas.canvas, (15, position), 1, (0, 180, 0), 3)

            cv2.putText(canvas.canvas,
                        str(dot),
                        (0, position),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)

        step = 15
        i = 1
        while i < len(avg_leghts):
            cv2.line(canvas.canvas,
                     (step*i, avg_leghts[i-1]),
                     (step*i+step, avg_leghts[i]),
                     (0, 0, 255),
                     2)
            i += 1

        cv2.line(canvas.canvas,
                 (15, avg_leghts[-1]),
                 (i*step, avg_leghts[-1]),
                 (0, 120, 0),
                 1)

        cv2.arrowedLine(canvas.canvas,
                        (0+600, 0+150),
                        (int(grid.global_direction_vector[0]/8) + 600,
                         int(grid.global_direction_vector[1]/8) + 150),
                        (0, 0, 0),
                        2)

        self.show_canvas(win, canvas)
