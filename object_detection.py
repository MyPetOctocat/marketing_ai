import cv2
import numpy as np
import copy
import time
import math



"""Initialisierung"""


net = cv2.dnn.readNet('yolov4-tiny_final.weights', 'yolov4-tiny.cfg')    # yolo Modell laden
#net = cv2.dnn.readNet('yolov3-tiny_training_final.weights', 'tiny-yolo_testing.cfg')    # yolo Modell laden
classes = ["hand", "long", "round", "square"]
objects = [["hand"], ["spaghettoni", "spaghettini"], ["pomodoro"], ["girandole", "farfalle"]]
cap = cv2.VideoCapture("Test v5.mp4")   # Videoinput (Webcam oder mp4)

# Farbkodierung nach Kategorie
color_square = [255, 0, 0]
color_round = [0, 255, 0]
color_long = [0, 255, 255]
color_hand = [255, 255, 0]
color_list = [color_square, color_round, color_long, color_hand]
font = cv2.FONT_HERSHEY_PLAIN   #Schriftart

### Initialisierung der Variablen für Homography
orb = cv2.ORB_create()  # Feature Detector initialisieren
## Referenzbilder der Produkte für Feature Matching
# square
girandole_img = cv2.imread(r"pics/girandole.jpg")
farfalle_img = cv2.imread(r"pics/farfalle.jpg")
# round
pomodoro_img = cv2.imread(r"pics/pomodoro.jpg")
# long
spaghettini_img = cv2.imread(r"pics/spaghettini_font.jpg")
spaghettoni_img = cv2.imread(r"pics/spaghettoni_font.jpg")

#Berechnung der Features der Referenzbilder (Trainingsbilder)
# square
girandole_kp, girandole_des = orb.detectAndCompute(girandole_img, None)
farfalle_kp, farfalle_des = orb.detectAndCompute(farfalle_img, None)
# round
pomodoro_kp, pomodoro_des = orb.detectAndCompute(pomodoro_img, None)
# long
spaghettoni_kp, spaghettoni_des = orb.detectAndCompute(spaghettoni_img, None)
spaghettini_kp, spaghettini_des = orb.detectAndCompute(spaghettini_img, None)
##Liste mit Features jedes Produkts
objects_kp = [[], [spaghettoni_kp, spaghettini_kp],[],[girandole_kp, farfalle_kp]]
objects_des = [[], [spaghettoni_des, spaghettini_des], [], [girandole_des, farfalle_des]]

# Feature Matcher um Frame mit Referenzen zu vergleichen
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

### Handbewegungen
pos = list()    # Koordinaten wo im Regal reingegriffen wurde
latest_pos = None   # Position der Hand
latest_barrier_pos = None   # Position der Hand, wenn Schranke durchschritten
hand_size = None
barrier = 600   # Schranke --> markiert Grenze zum Regal
hand_pos = None
hand_level = None
product_label = None
hand_coordinates = [100000, 1000000]
width_ratio = 0


### Für die Auswertung & Ergebnisse ###
counter = 0 # Framezähler zum Berechnen von Durchschnittswerten
hand_counter = 0

#[["hand"], ["spaghettoni", "spaghettini"], ["pomodoro"], ["girandole", "farfalle"]]
product = objects[2][0] # Zu klassifizierendes Produkt (Adresse angeben)
product_category = objects[2]   # Klasse des zu klassifizierenden Produkts
# Notierung der erkannten Objekte
detection_results = np.array([0, 0, 0, 0]) # [Right Product, Right Category, False Category, No Detection]
# fps
fps_sum = 0 # Summe der fps pro Frame um später durchschnittliche fps zu ermitteln
# Statistik
test_range = [405, 415] # Der Zeitraum in den der Test durchgeführt werden soll (Videoabschnitt wo die Erkennung gemessen werden soll)
label_list = [] # alle Erkannten Labels in diesem Zeitraum werden in dieser Liste gespeichert


### Shelf
# Erstellung eines Regalmodells
shelf = np.zeros((450, 450, 3), np.uint8) #gewünschte Auflösung der Shelf
shelf[:] = (255, 255, 255)    # Weißer Hintergrund als Basis
shelf_height, shelf_width = shelf.shape[:2]
# Vertikale Linien
cv2.line(shelf, (int((1/3)*shelf_width)+15, 0), (int((1/3)*shelf_width + 40)+15, shelf_height), [255, 255, 200], 2)
cv2.line(shelf, (int((2/3)*shelf_width)+15, 0), (int((2/3)*shelf_width - 40)+15, shelf_height), [255, 255, 200], 2)
# Horizontale Linien
cv2.line(shelf, (65, int((4/10)*shelf_height)), (shelf_width-25, int((4/10)*shelf_height)), [0, 200, 220], 2)
cv2.line(shelf, (110, int((3/4)*shelf_height)), (shelf_width-80, int((3/4)*shelf_height)), [0, 200, 220], 2)
# Ränder des Regals (Aufgrund der Perspektive)
cv2.line(shelf, (int((0)*shelf_width)+15, 0), (int((2/7)*shelf_width)+15, shelf_height), [0, 0, 0], 3)
cv2.line(shelf, (int((1)*shelf_width + 20)+15, 0), (int((5/7)*shelf_width)+15, shelf_height), [0, 0, 0], 3)
shelf_clean = copy.deepcopy(shelf)  # leeres Regal für Resets in der Visualisierung
# Erstellung eines Produkticons, welches auf dem virtuellen Regal plaziert wird
comp_mapping = [int((4/10)*shelf_height-105), int((3/4)*shelf_height-80), int(shelf_height)-55] # Höhe des jeweiligen Compartment des Regals: Idx = 0 --> Oberstes Regal ; 1 --> Mittleres Regal
product_size = [100, 75, 50]    # Größe des Produkts (abhängig von der Regalhöhe (Perspektive --> unterstes Regal == kleinstes Bild; oberstes Regal == größtes Bild


def create_shelf_product(product_img, x, shelf_level):
    size = product_size[shelf_level]
    y_offset = comp_mapping[shelf_level] - 25
    x_offset = x
    product_img = cv2.resize(product_img, (size, size))
    shelf[y_offset:y_offset + product_img.shape[0], x_offset:x_offset + product_img.shape[1]] = product_img
    return [product_img, x_offset, y_offset]

### Regalaufstellung
# Produkte oberes Regal
product_img1 = product_img = cv2.imread(r"pics/girandole.jpg")
product_img2 = product_img = cv2.imread(r"pics/spaghettoni.jpg")
product_img3 = product_img = cv2.imread(r"pics/pomodoro.jpg")
# Produkte mittleres Regal
product_img4 = product_img = cv2.imread(r"pics/pomodoro.jpg")
product_img5 = product_img = cv2.imread(r"pics/girandole.jpg")
product_img6 = product_img = cv2.imread(r"pics/spaghettoni.jpg")

product_list = [create_shelf_product(product_img1, 325, 0),
                create_shelf_product(product_img2, 200, 0),
                create_shelf_product(product_img3, 80, 0),
                create_shelf_product(product_img4, 280, 1),
                create_shelf_product(product_img5, 200, 1),
                create_shelf_product(product_img6, 120, 1),
                ]

alpha = 0.3 # Transparenzwerte für die Handpunkte
shelf_display = shelf   # Regal Version, welche später ausgegeben werden soll
# Statusvariablen
empty_hands = True  # Wird ein Produkt in der Hand gehalten?
enter_empty = False  # Wird auf True gesetzt, wenn Schranke ohne Produkt überschritten wurde (True bis Schranke wieder verlassen wurde)
enter_full = False     # Wird auf True gesetzt, wenn Schranke mit Produkt überschritten wurde
lock = False

#hand_list = []


"""Erkennung"""

while True:

    counter += 1 # Frame Counter wird befüllt

    start = time.time()
    shelf = shelf_display
    _, frame = cap.read() # Frame für die Klassifizierung extrahieren
    _, show_frame = cap.read() # Frame für die Visualisierung & Informationen extrahieren
    try:    # Wenn es keine Frames mehr gibt, wird die while Schleife unterbrochen
        height, width, _ = frame.shape #Auflösung des Frames, um Linien zu setzen
    except AttributeError:
        break
    cv2.rectangle(frame, (0,0), (750, 350), (0,0,0), -1) # Unkommentieren, wenn 2 Hände im Video zu sehen sind, sonst gibt es Probleme
    cv2.rectangle(frame, (0,0), (1000, 80), (0,0,0), -1) # Unkommentieren, wenn 2 Hände im Video zu sehen sind, sonst gibt es Probleme

    cv2.line(show_frame, (0, barrier), (width, barrier), [255, 255, 0], 2)  # Markiert die Grenze zum Regal
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names) # Ergebnis/Output jedes grids
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs: # Iteration über jede Gridwerte
        for detection in output:
            scores = detection[5:]  # Array mit Confidence aller Labels
            class_id = np.argmax(scores)    # Index des Labels mit der höchsten Confidence wird genommen
            confidence = scores[class_id]
            if confidence > 0.5:    # Detection nur von Interesse wenn die Konfidenz höher 0.5 ist (um Noise rauszufiltern)
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])  # Koordinaten der BoundingBox(BB)
                confidences.append(float(confidence))
                class_ids.append(class_id)  # Liste mit den Indices des Labels gespeichert: int 0 = square, int 1 = round, int 2 = long, int 3 = hand
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    # Non-Max-Suppression: Im Bereich mit hoher Detection wird die BB mit der höchsten Confidence genommen; Die Indices aller endgültig entschiedener Objekte werden in die Variabele gespeichert
    if len(indices) > 0:    #Kategorie wurde registriert
        for i in indices.flatten():
            if class_ids[i] == 0 and len(indices) > 1:  # Wenn eine Hand und ein weiteres Produkt erkannt wurde...
                empty_hands = False # ...wird davon ausgegangen, dass dieses Produkt in der Hand gehalten wird
            elif class_ids[i] == 0 and len(indices) == 1:   # Wenn sich nur die Hand im Bild befindet...
                empty_hands = True
        for i in indices.flatten(): # Iteration über jedes Erkannte Objekt
            x, y, w, h = boxes[i]   # BB Koordinaten entnommen
            x, y, w, h = max(x, 0), max(y, 0), min(w, w + x), min(h, h + y)     # Für den Fall, dass BB über das Window hinausgeht
            classes_index = class_ids[i]    # Die Kategorie des Objektes wird in einer Variable gespeichert
            # Produktunterscheidung mit Homography
            if len(objects[classes_index]) > 1:     # Wenn es sich um keine Hand handelt, oder eine Kategorie mehr als ein Produkt beinhaltet --> Feature Matching...
                crop_img = frame[y: y + h, x: x + w, :] # BB wird aus dem Frame ausgeschnitten und in eine Variable zum Feature Matching gespeichert
                crop_kp, crop_des = orb.detectAndCompute(crop_img, None) # Feature Detection mit dem Produkt
                distances = list()  # Die Distanzen (Ähnlichkeiten zwischen dem Query Image [Referenzbild] und dem Training Image [BB des Frames]) für jedes Objekt
                distance_score = list() # Durschnittliche Distanz der Query Images zum Frame
                for j in range(len(objects[classes_index])):
                    distances.append(list())
                    matches = bf.match(crop_des, objects_des[classes_index][j]) # Features des Query und Training Images werden auf Ähnlichkeiten geprüft
                    matches = sorted(matches, key=lambda x:x.distance)  # Features werden nach ihrer Distanz sortiert
                    top_matches = matches[:20]  # Nur die 20 Ähnlichsten Feature sollen weiter betrachtet werden (um Noise zu unterdrücken)
                    for m in top_matches:
                        distances[j].append(m.distance) # 20 Distanzen werden in Liste gespeichert
                    distance_score.append(sum(distances[j])/20) # Durschnittswert der 20 Distanzen (um zu unterscheiden, welches Produkt dem gecroppten Frame zuzuordnen ist)
                object_index = np.argmin(distance_score)    #Index des wahrscheinlichsten Produkts
                label = objects[classes_index][object_index]   # Das erkannte Produkt
                confidence = str(round(confidences[i], 2)) # Zuversichtlichkeit über die Kategorie
            else:
                label = objects[classes_index][0]    # ...ansonsten das einzelne Produkt mit Index 0 zurückgeben
                confidence = str(round(confidences[0], 2)) # Zuversichtlichkeit über das ausgewählte Label

            #Visualisierung
            color = color_list[class_ids[i]] # Farbe der BB
            cv2.rectangle(show_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(show_frame, "{} ({} {})".format(label, confidence, classes[classes_index]), (x, y + 20), font, 2, (0, 0, 0), 2)
            # Hand-Produkt-Interaktion
            if classes_index != 0:
                if counter >= test_range[0] and counter <= test_range[1]:
                    label_list.append([label, classes_index])
                if y+(h/2) <= barrier:  #label des Produkts, welches in der Hand gehalten wird, wird in Variable gespeichert, bis es die Schranke durchbricht
                    product_label = label
            if classes_index == 0:
                hand_counter += 1   # zählt hoch, sobald hand im Frame zu sehen
                latest_pos = [copy.deepcopy(x), copy.deepcopy(w), copy.deepcopy(y), copy.deepcopy(h)]   # Damit die Koordinaten von der Hand nicht überschrieben werden (keine Referenz auf x), muss eine DeepCopy erstellt werden
                pos.append(latest_pos)
                if y+(h/2) >= barrier:    # Wenn Hand die Regalgrenze überschreitet, wird das als "reingreifen" registriert (mittlere Hanhöhe als Punkt gewählt)
                    if empty_hands == True and lock == False: # Wenn Schranke ohne Produkt durchbrochen --> passed_empty = True und halte diesen Wert, bis Schranke wieder verlassen wird
                        enter_empty = True
                        lock = True
                    elif empty_hands == False and lock == False:                   # ... sonst wurde die Schranke mit Produkt in der Hand durchbrochen
                        enter_full = True
                        lock = True
                    latest_barrier_pos = [copy.deepcopy(x), copy.deepcopy(w)]
                    # Shelf
                    width_ratio = shelf_width/width   # Skalierung von Video auf Shelfmodell
                    height_ratio = shelf_height/height    # Beachten height im Video != height im Shelf --> Verschiedene Perspektiven
                    overlay = shelf.copy()  # Shelf kopieren zum Erstellen von Layers für alpha-transparency Effekte
                    overlay_clean = shelf_clean.copy()  # Dasselbe wird mit dem leeren Shelf gemacht
                    hand_coordinates = (int((x+w/2)*width_ratio), hand_pos) #Handkoordinaten, wie diese auf dem virtuellen Regal angezeigt werden
                    cv2.circle(overlay, hand_coordinates, int((w/2)*width_ratio), (0, 0, 255), -1) # Handposition im Regal wird mit einem roten Punkt markiert
                    cv2.circle(overlay_clean, hand_coordinates, int((w/2)*width_ratio), (0, 0, 255), -1)
                    shelf = cv2.addWeighted(overlay, alpha, shelf, 1 - alpha, 0)    # Layer wird dem Shelf hinzugefügt
                    shelf_clean = cv2.addWeighted(overlay_clean, alpha, shelf_clean, 1 - alpha, 0)
                    shelf_display = shelf   # Version des Shelfs, welche dargestellt wird
                else:
                    hand_size = round(math.sqrt(latest_pos[1] * latest_pos[3]), 3)  # Bestimmung der Größe der Handfläche
                    if hand_size > 215: # Wenn die BB der Hand (vor kurz vor überschreiten der Schranke größer als *Wert* ist, dann greift die Person am oberen Regal
                        hand_level = 0
                        hand_pos = comp_mapping[0]
                    else:               # Sonst greift die Person am unteren Regal
                        hand_level = 1
                        hand_pos = comp_mapping[1]
                    lock = False    # Wenn die Schranke wieder verlassen wurde, resette den Lock
                    for i in range(len(product_list)):
                        product_img = product_list[i][0]
                        x_offset = product_list[i][1]   # Koordinaten des Produkts auf Variable speichern, sonst stürzt es ab eine line drunter
                        y_offset = product_list[i][2]
                        product_height = product_img.shape[0]
                        product_width = product_img.shape[1]
                        hand_x = hand_coordinates[0]
                        #print(hand_x - (x_offset + product_width/2))
                        #print(int(w/2) * width_ratio)
                        #print(abs(hand_x - x_offset + product_width/2))
                        print(hand_x - (product_list[3][1] + product_width/2))
                        print(hand_x)
                        if enter_empty and empty_hands == False and (abs(hand_x - (x_offset + product_width/2))) < int(w/2) * width_ratio and hand_pos-25 == y_offset: # Wenn ohne Produkt ins Regal gefasst wurde und beim Verlassen ein Produkt in der Hand liegt (und in die Region gefasst wurde)...
                            enter_empty = False
                            print("Produkt genommen")
                            #Visualisierung
                            cv2.rectangle(shelf_display, (x_offset, y_offset), (x_offset+product_height, y_offset+product_width), (255, 255, 255), -1)  # Unkommentieren, wenn 2 Hände im Video zu sehen sind, sonst gibt es Probleme
                            cv2.putText(shelf_display, "TAKEN",(x_offset, y_offset+50), font, 2, (0, 255, 0), 2)
                        elif enter_full and empty_hands == True:    # Wenn die Hand die Schranke mit einem Produkt in der Hand durchschritten hat und mit leeren Händen das Regal verlässt...
                            enter_full = False
                            print("Produkt zurückgelegt")
                            if x_offset < 0: # Verlässt das Produkt die Boundary des Shelfs...
                                x_offset = 0 #... wird das Produkt an den Rand gesetzt
                            else:
                                x_offset = min((shelf_width-100)+product_img.shape[1], (hand_coordinates[0]-50)+product_img.shape[1])    # Verlässt das Produkt die Boundary des Shelfs wird das Produkt an den Shelf Rand gesetzt
                                x_offset -= product_img.shape[1]
                                print(product_label)
                                product_img = cv2.imread(r"pics/{}_thumbnail.jpg".format(product_label)) # Das Produkt welches gesehen wurde bevor die Schranke durchschritten wurde, wird als "zurückgelegt visualisiert"
                            create_shelf_product(product_img, x_offset, hand_level)
    else:
        detection_results[3] += 1   # Kein Produkt wurde in diesem Frame erkannt
    if latest_barrier_pos != None:  # Wenn Schranke überschritten wurde
        cv2.line(show_frame, (latest_barrier_pos[0], barrier), ((latest_barrier_pos[0] + latest_barrier_pos[1]), barrier), [0, 0, 255], 2)   # Farbliche Markierung der Stelle in die reingegriffen wurde
        cv2.putText(show_frame, "x:{} y:{}".format(latest_barrier_pos[0] + latest_barrier_pos[1] / 2, barrier), (20, 50), font, 1, (0, 0, 255), 2)  # Koordinaten in schriftlicher Form
    if latest_pos != None:
        cv2.putText(show_frame, "x:{} y:{}".format(latest_pos[0] + (latest_pos[1] / 2), barrier), (20, 100), font, 1, (0, 255, 255), 2)  # Koordinaten in schriftlicher Form
    # fps-Zähler
    end = time.time()
    duration = end - start
    fps = round(1/duration, 2)
    cv2.putText(show_frame, "fps: {}".format(fps), (20, 150), font, 2, (255, 255, 255), 2)  # fps Anzeige
    fps_sum += fps

    # Ausgabe skalieren
    scale = 0.5
    try:
        show_frame = cv2.resize(show_frame, (int(show_frame.shape[1]*scale), int(show_frame.shape[0]*scale)))
        frame = cv2.resize(frame, (int(show_frame.shape[1]*scale), int(show_frame.shape[0]*scale)))
        cv2.imshow("Video", show_frame)
        cv2.imshow("Video_Raw", frame)
        cv2.imshow("Shelf", shelf_display)

    except:
        pass
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
print()

print(label_list)

print("frames:\t" + str(counter))

# durchschnittliche Handgröße
hand_sum = 0
for i in range(len(pos)):
    w = pos[i][1]
    h = pos[i][3]
    hand_sum += math.sqrt(w*h)

print("Durchschnittliche Handgröße" + str(hand_sum/hand_counter))
print("Durchschnittliche FPS:" + str(fps_sum/counter))


print("EVALUATION\n")

print(test_range[1]-test_range[0])
print(len(label_list))

# Für Auswertung auskommentieren
# detection_results[3] = (test_range[1]-test_range[0]+1) - len(label_list)  # --> Frames in denen kein Produkt erkannt wurde
# for label in label_list:
#     product_name = label[0]
#     product_class = label[1]
#     print(product_name, product)
#     if product_name == product:
#         detection_results[0] += 1   # Wenn das richtige Produkt erkannt wurde
#     elif product_name in product_category:
#         #print(objects[product_class])
#         detection_results[1] += 1   # Wenn das falsche Produkt aber die Kategorie erkannt wurde
#     else:
#         detection_results[2] += 1   # Wenn das falsche Produkt aus einer anderen Kategorie erkannt wurde
#
# print(
#     "Richtiges Produkt:\t{}\n"
#     "Richtige Kategorie:\t{}\n"
#     "Falsche Kategorie:\t{}\n"
#     "Kein Produkt:\t\t{}\n"
#     .format(detection_results[0], detection_results[1], detection_results[2], detection_results[3])
# )


cv2.imshow("Customer Tracking", shelf_clean)
cv2.waitKey(0)