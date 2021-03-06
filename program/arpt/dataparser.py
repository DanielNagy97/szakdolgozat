import json
import os

from arpt.canvas import Canvas
from arpt.window import Window

from arpt.widget import Widget
from arpt.widgets.shiftable import Shiftable
from arpt.widgets.expandable import Expandable
from arpt.widgets.button import Button
from arpt.widgets.grabbable import Grabbable
from arpt.widgets.tuner import Tuner
from arpt.widgets.rollable import Rollable


class DataParser(object):
    def __init__(self):
        pass

    def read_json(self, project_path, file_path):
        """
        Read Json file
        :param project_path: The path of the project's folder
        :param file_path: The path of the file in the project's folder
        :return: data as dictionary
        """
        with open(os.path.join(project_path, file_path), 'r') as cfile:
            data = json.load(cfile)
        return data

    def build_canvasses_from_pref(self, video, path):
        """
        Setting up windows according to preferences file
        :param path: The path of the project's folder
        :return: Canvasses as dictionary
        """
        canvasses_data = self.read_json(path, 'preferences/canvasses.json')

        canvasses = dict()

        for canvas in canvasses_data:
            if canvas['dimension'] == "video":
                dimension = video.dimension
            else:
                dimension = tuple(canvas['dimension'])

            if 'fill' in canvas:
                canvasses[canvas['name']] = Canvas(dimension,
                                                   canvas['channels'],
                                                   canvas['fill'])
            else:
                canvasses[canvas['name']] = Canvas(dimension,
                                                   canvas['channels'])
        return canvasses

    def build_windows_from_pref(self, path):
        """
        Setting up windows according to preferences file
        :param path: The path of the project's folder
        :return: Windows as dictionary
        """
        windows_data = self.read_json(path, 'preferences/windows.json')

        windows = dict()

        for window in windows_data:
            windows[window['name']] = Window(window['title'],
                                             window['mode'],
                                             tuple(window['position']))
            if 'dimension' in window:
                windows[window['name']].resize(*window['dimension'])

        return windows

    def build_scene_from_project_file(self, path):
        """
        Building the scene from project file
        :param path: The path of the project's folder
        :return: Scene as list of objects of dicts
        """
        scene_data = self.read_json(path, 'scene.json')

        scene = []

        for i in range(len(scene_data)):
            slide = []
            for widget in scene_data[i]['widgets']:
                if widget['type'] == 'Button':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Button(tuple(widget['position']),
                                          tuple(widget['dimension']),
                                          image_path,
                                          widget['transparent'],
                                          widget['action'])
                elif widget['type'] == 'Tuner':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Tuner(tuple(widget['position']),
                                         tuple(widget['dimension']),
                                         image_path,
                                         widget['min_value'],
                                         widget['max_value'],
                                         widget['transparent'],
                                         widget['action'])
                elif widget['type'] == 'Shiftable':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Shiftable(tuple(widget['position']),
                                             tuple(widget['dimension']),
                                             image_path,
                                             widget['speed'],
                                             widget['attenuation'],
                                             widget['transparent'])
                elif widget['type'] == 'Expandable':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Expandable(tuple(widget['position']),
                                              tuple(widget['dimension']),
                                              image_path,
                                              widget['speed'],
                                              widget['attenuation'],
                                              widget['header_size'],
                                              widget['transparent'])
                elif widget['type'] == 'Grabbable':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Grabbable(tuple(widget['position']),
                                             tuple(widget['dimension']),
                                             image_path,
                                             widget['transparent'])
                elif widget['type'] == 'Rollable':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Rollable(tuple(widget['position']),
                                            tuple(widget['dimension']),
                                            image_path,
                                            widget['speed'],
                                            widget['attenuation'],
                                            tuple(widget['box_dimension']),
                                            widget['transparent'])
                elif widget['type'] == 'Widget':
                    image_path = os.path.join(path, widget['image'])
                    widget_class = Widget(tuple(widget['position']),
                                          tuple(widget['dimension']),
                                          image_path,
                                          widget['transparent'])

                slide.append(widget_class)
            scene.append({'widgets': slide})

        return scene
