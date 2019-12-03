# Background Substraction Trial

Ezen rövig script az openCV-ben található *Background Substraction* függvények a kipróbálására jött létre.
`cv2.createBackgroundSubtractorMOG2` `cv2.createBackgroundSubtractorKNN`
* Ezen eljárások adaptív háttér-tanulásos tecnikákat alkalmaznak, futás közben számítják ki, hogy mely képrészek tatoznak a háttérhez és melyek az előtérhez. A háttérmodell folyamatosan frissül, ezáltal az olyan elemek, amelyek kevésbé mozognak, idővel a háttérmodell részévé válhatnak.
[Bővebb leírás az openCV backgroung substraction módszereiről](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html).

* [BackgroundSubtractorGMG](https://docs.opencv.org/4.1.1/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html) Ez az eljárás "előtanuló" technikát alkalmaz. Az első n képkockát felhasználva megalkot egy statikus háttérmodellt és ennek a segítségével határozza meg, hogy mi tartozik a háttérhez és mi nem. Ez a függvény nem része az OpenCV 4.1.1-nek. Az *opencv-contrib-python* **UNOFFICIAL** csomaggal telepíthető.