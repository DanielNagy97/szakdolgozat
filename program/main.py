"""
Technical demonstration of the ARPT library
"""
import sys
from arpt.controller import Controller

if __name__ == "__main__":
    # print(sys.argv)
    # project_path = "./projects/dolgozat/"
    project_path = sys.argv[1]
    mode = sys.argv[2]
    if mode.lower() in ['true', '1', 't', 'y', 'yes', 'igen']:
        demo = True
    else:
        demo = False

    controller = Controller(project_path, demo)
    controller.main_loop()
