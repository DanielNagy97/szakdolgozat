import cv2
import numpy

width, height = 512, 512

canvas = numpy.zeros((width,height,3), numpy.uint8)

x = 200
y = 150
w = 100
h = 100

oldX = 144
oldY = 141

newX = 250
newY = 200

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return int(px), int(py)

while True:

    cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.arrowedLine(canvas, (oldX, oldY), (newX, newY), (0,0,255), 2)

    if x < oldX < x+w and y < oldY < y+h:
        #print("a regi benne van")
        pass
    else:
        if x < newX < x+w and y < newY < y+h:
            #print("benne")

            if oldX < x and y < oldY < y+h:
                #print("1")
                interx, intery = findIntersection(oldX,oldY,newX,newY,x,y,x,y+h)
            if x < oldX < x+w and oldY < y:
                #print("2")
                interx, intery = findIntersection(oldX,oldY,newX,newY,x,y,x+w,y)
            if  x < oldX < x+w and oldY > y+h:
                interx, intery = findIntersection(oldX,oldY,newX,newY,x,y+h,x+w,y+h)
                #print("3")
            if oldX > x+w and y < oldY < y+h:
                interx, intery = findIntersection(oldX,oldY,newX,newY,x+w,y,x+w,y+h)
                #print("4")

            if oldX < x and oldY < y:
                interx, intery = findIntersection(oldX,oldY,newX,newY,0,y,x,y)
                if interx < x:
                    interx, intery = findIntersection(interx,intery,newX,newY,x,y,x,y+h)
            if oldX < x and oldY > y+h:
                interx, intery = findIntersection(oldX,oldY,newX,newY,0,y+h,x,y+h)
                if interx < x:
                    interx, intery = findIntersection(interx,intery,newX,newY,x,y,x,y+h)
            if oldX > x+w and oldY < y:
                interx, intery = findIntersection(oldX,oldY,newX,newY,width,y,x+w,y)
                if interx > x+w:
                    interx, intery = findIntersection(interx,intery,newX,newY,x+w,y,x+w,y+h)
            if oldX > x+w and oldY > y+h:
                interx, intery = findIntersection(oldX,oldY,newX,newY,width,y+h,x+w,y+h)
                if interx > x+w:
                    interx, intery = findIntersection(interx,intery,newX,newY,x+w,y,x+w,y+h)

            cv2.circle(canvas, (interx,intery), 5, (0, 0, 255), 2)     

            dvX = newX-interx
            dvY = newY-intery
            x = x+dvX
            y = y+dvY

        else:
            #print("nincs benne")
            pass

    cv2.imshow("Test",canvas)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
            break
cv2.destroyAllWindows()