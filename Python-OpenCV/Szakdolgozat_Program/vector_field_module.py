import cv2
import numpy as np


def draw_vector_field(old_points,new_points,vector_field_canvas):
    vector_field_canvas.fill(255)
    for k in range(int(new_points.size/2)):
        current_vector = np.subtract(new_points[k],old_points[k])
        if abs(current_vector[0]) >= 2 or abs(current_vector[1]) >= 2:
            cv2.arrowedLine(vector_field_canvas, tuple(old_points[k]), tuple(new_points[k]), (0,0,0), 2)


def heat_map(old_points, new_points, heat_map_canvas):
    old_points_3D = old_points.reshape(8,15,2)
    new_points_3D = new_points.reshape(8,15,2)

    vector_lenghts = np.sqrt(np.sum(np.power(np.subtract(new_points,old_points),2),axis=1))
    heat_values = np.int32(np.multiply(vector_lenghts,10))
    heat_values = np.where(heat_values > 255,255,heat_values)

    heat_map_canvas = np.uint8(np.dstack((np.subtract(255,heat_values),np.zeros(len(heat_values),dtype=np.uint8),heat_values)).reshape(8,15,3))

    resized_heat_map = cv2.resize(heat_map_canvas, dsize=(600, 320), interpolation=cv2.INTER_AREA)

    gray_heat_map = cv2.cvtColor(heat_map_canvas,cv2.COLOR_BGR2GRAY)
    ret, thresholded_heat = cv2.threshold(gray_heat_map,40,255,cv2.ADAPTIVE_THRESH_MEAN_C)
    contours, hierarchy = cv2.findContours(thresholded_heat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    different_direction = 0

    motion_points = np.empty((0,2), dtype=np.float32)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        rectArea = w*h
        if rectArea < 2:
            continue
        else:
            count += 1
            local_vector_sum = np.array([old_points_3D[y:y+h,x:x+w].sum(axis=0),new_points_3D[y:y+h,x:x+w].sum(axis=0)], dtype=np.float32).sum(axis=1)
            local_direction_vector = np.subtract(local_vector_sum[1],local_vector_sum[0])

            local_normalized_direction_vector = normalize_vector(local_direction_vector)

            cv2.rectangle(resized_heat_map, (x*40,y*40), (x*40+w*40, y*40+h*40), (255,255,255),2)
            cv2.arrowedLine(resized_heat_map, (int(x*40+(w*40)/2), int(y*40+(h*40)/2)), (int(local_normalized_direction_vector[0]*100+x*40+(w*40)/2), int(local_normalized_direction_vector[1]*100+y*40+(h*40)/2)), (0,255,255), 2)

            motion_points = np.append(motion_points, np.array([local_normalized_direction_vector]), axis=0)

    if len(motion_points) == 2:
        motion_points_sum = np.abs(np.sum(motion_points,axis=0))
        sum_of_a_sum = np.sum(motion_points_sum)

        epsilon = 0.9

        different_direction = (epsilon-sum_of_a_sum)*100

        if different_direction < 0:
            different_direction = 0
    
    cv2.putText(resized_heat_map, str(np.format_float_positional(different_direction,precision=2)), (10,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,0), 3, cv2.LINE_AA)
    #cv2.putText(resized_heat_map, str(count), (10,150), cv2.FONT_HERSHEY_SIMPLEX , 5, (0,0,0), 3, cv2.LINE_AA)

    cv2.imshow("HeatMap",resized_heat_map)
    #cv2.imshow("HeatMap",heat_map_canvas)
    #cv2.imshow("ThresholdedHeatMap",thresholded_heat)


avg_vector_lenghts=[]

def global_resultant_vector(old_points,new_points,plot_canvas):
    global avg_vector_lenghts

    vector_sum = np.array([old_points.sum(axis=0),new_points.sum(axis=0)], dtype=np.float32)
    vector_count = int(old_points.size/2)

    global_direction_vector = np.subtract(vector_sum[1],vector_sum[0])
    global_direction_vector_length = get_vector_lenght(global_direction_vector)
    average_vector_lenght = global_direction_vector_length/vector_count
    avg_vector_lenghts.append(average_vector_lenght)

    avg_vector_lenghts = avg_vector_lenghts[-30:]

    show_global_results(avg_vector_lenghts,global_direction_vector,plot_canvas)


def get_vector_lenght(vector):
    return np.sqrt(np.sum(np.power(vector,2)))

def normalize_vector(vector):
    return np.divide(vector, get_vector_lenght(vector)) 

def show_global_results(avg_vector_lenghts,global_direction_vector,canvas):
    canvas.fill(255)

    cv2.putText(canvas, 'Global Resultant Vector', (250,15), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'AVG Vector Lenght', (0,15), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.line(canvas, (15, 285), (470, 285), (0,180,0), 1)
    cv2.line(canvas, (15, 285), (15, 20), (0,180,0), 1)

    cv2.putText(canvas, 'Direction', (560,40), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.line(canvas, (600, 250), (600, 50), (0,180,0), 1)
    cv2.line(canvas, (500, 150), (700, 150), (0,180,0), 1)

    avg_leghts = np.int32(np.add(np.multiply(avg_vector_lenghts,-20),285))
    
    dots = [0,5,10]

    for dot in dots:
        position = np.int32(np.add(np.multiply(dot,-20),285))
        cv2.circle(canvas,(15,position),1,(0,180,0),3)
        cv2.putText(canvas, str(dot), (0,position), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
   
    step = 15
    i = 1
    while(i<len(avg_leghts)):
        cv2.line(canvas, (step*i,avg_leghts[i-1]), (step*i+step,avg_leghts[i]), (0,0,255), 2)
        i+=1
    cv2.line(canvas,(15,avg_leghts[-1]),(i*step,avg_leghts[-1]),(0,120,0),1)

    cv2.arrowedLine(canvas, (0+600, 0+150), (int(global_direction_vector[0]/8)+600, int(global_direction_vector[1]/8)+150), (0,0,0), 2)