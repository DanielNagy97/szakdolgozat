# Background Substraction Trial

Ezen rövig script az openCV-ben található *Background Substraction* függvények a kipróbálására jött létre.
`cv2.createBackgroundSubtractorMOG2` `cv2.createBackgroundSubtractorKNN`
* Ezen eljárások adaptív háttér-tanulásos tecnikákat alkalmaznak, futás közben számítják ki, hogy mely képrészek tatoznak a háttérhez és melyek az előtérhez. A háttérmodell folyamatosan frissül, ezáltal az olyan elemek, amelyek kevésbé mozognak, idővel a háttérmodell részévé válhatnak.
[Bővebb leírás az openCV backgroung substraction módszereiről](http://www.uni-miskolc.hu/~qgenagyd/references/OpenCVBackgroundSubstraction/Background%20Subtraction%20%e2%80%94%20OpenCV-Python%20Tutorials%201%20documentation.html).

* [BackgroundSubtractorGMG](http://www.uni-miskolc.hu/~qgenagyd/references/OpenCVBSGMG/OpenCV%20%20cv%20bgsegm%20BackgroundSubtractorGMG%20Class%20Reference.html) Ez az eljárás "előtanuló" technikát alkalmaz. Az első n képkockát felhasználva megalkot egy statikus háttérmodellt és ennek a segítségével határozza meg, hogy mi tartozik a háttérhez és mi nem. Ez a függvény nem része az OpenCV 4.1.1-nek. Az *opencv-contrib-python* **UNOFFICIAL** csomaggal telepíthető.