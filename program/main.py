"""
Technical demonstration of the ARPT library
"""

from arpt.controller import Controller

if __name__ == "__main__":
    controller = Controller("./projects/presentation_01/")
    controller.main_loop()
