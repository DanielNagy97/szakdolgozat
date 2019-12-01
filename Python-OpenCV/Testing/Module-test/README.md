# Module Testing Trial

Az eddig kipróbált eljárások, megoldások csokorba szedése.


## vectorFieldModule.py

A modulban található eljárások bővebb leírása [**itt**](https://github.com/DanielNagy97/szakdolgozat/tree/master/Python-OpenCV/Testing/Lucas-Kanade) érhető el.

A modulban található metódusok:

caclOpticalFlow(oldGrayFrame,grayFrame,oldPoints)
- A `cv2.calcOpticalFlowPyrLK`-t hívja meg a kapott paraméterekkel. Az Optical Flow további paraméterei a modulban vannak definiálva

vectorFieldGrid(gridStep, capWidth, capHeight)
- Elkészíti a pontok halmazát, amelyek segítségével a vektormezőt megvalósíthatjuk.
    
drawVectorField(canvas,oldPoints,newPoints)
- Kirajzolja a vektormezőt a `cv2.arrowedLine` primitívrajzoló segítségével.

getVectorLength(vector):
- Visszaadja a paraméterként megadott vektor hosszát.


## framediffModule.py

Ezen modulban egyetlen definíció található.

frameDifferencing(grayFrame,oldGrayFrame,canvas)
- Működése megegyezik a [**Frame Differencing trial**](https://github.com/DanielNagy97/szakdolgozat/tree/master/Python-OpenCV/Testing/FrameDifferencing)-ban leírt eljáráséval.
