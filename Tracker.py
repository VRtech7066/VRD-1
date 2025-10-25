'''Tracks Joints'''
import numpy as np
import cv2

'''------------------------------------------CLASS---------------------------------------'''
class Joint:
    # BGR
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    JOINT_COLOR = YELLOW
    BONECOLOR = GREEN
    DRAW_COLOR = RED

    # Window/image size
    Ncol = 800
    Nrow = 450
    # Defines how often points are sampled (1 in every __)
    sample = 10
    # defines how large an object must be to be recognized
    minContourSize = 75
    # Defines how poorly a circle can be drawn, and still be recognized as a circle
    circleStrayTollerance = 50
    circleClosedTollerance = 30
    circleMinRadius = 30
    # Line tolerances
    lineStrayTollerance = 30
    lineSpacingTollerance = 30
    lineMinVelocity = 5
    # Defines how still a joint must be, and still be recognized as still
    stillTollerance = 10

    # Import cascade for face recognition using OpenCV's built-in paths
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    fist_cascade = cv2.CascadeClassifier('/Users/veerraghuvanshi/Desktop/Veer/fist.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_glasses_cascade = cv2.CascadeClassifier('/Users/veerraghuvanshi/Desktop/Veer/haarcascade_eye_tree_eyeglasses.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
    full_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    @staticmethod
    def check_cascades():
        cascades = {
            "smile_cascade": Joint.smile_cascade,
            "fist_cascade": Joint.fist_cascade,
            "face_cascade": Joint.face_cascade,
            "face_glasses_cascade": Joint.face_glasses_cascade,
            "eye_cascade": Joint.eye_cascade,
            "upper_cascade": Joint.upper_cascade,
            "lower_cascade": Joint.lower_cascade,
            "full_cascade": Joint.full_cascade,
        }
        for name, cascade in cascades.items():
            if cascade.empty():
                print(f"Warning: {name} failed to load. Check the XML file path.")

    # Stores all the joints that have been created
    totalJointsList = []

    def __init__(self, x=0, y=0, colorName='no colorName', type='',
                 lowerRange=np.array([0, 0, 0]), upperRange=np.array([255, 255, 255]),
                 track=1, drawTrack=0, draw=1):
        self.x = x
        self.y = y
        self.rad = 10
        self.colorName = colorName
        self.type = type
        self.upperRange = upperRange
        self.lowerRange = lowerRange
        self.track = track
        self.draw = draw
        self.drawTrack = drawTrack
        self.state = ['', 50]
        self.jointTrail = []
        self.trailLength = 10

    def storeTrack(self):
        if len(self.jointTrail) < self.trailLength:
            if self.x != 0 or self.y != 0:
                self.jointTrail.append([self.x, self.y])
            elif len(self.jointTrail) > 0:
                self.jointTrail.pop(0)
        else:
            if len(self.jointTrail) > 0:
                self.jointTrail.pop(0)
            if self.x != 0 or self.y != 0:
                self.jointTrail.append([self.x, self.y])

    def still(self):
        jointTrail = self.jointTrail
        still = False
        if len(jointTrail) == self.trailLength:
            __, center = Joint.averagePoint(jointTrail)
            still = True

            for point in jointTrail:
                vector = [point[0] - center[0], point[1] - center[1]]
                magnitude = Joint.mag(vector)
                if magnitude > Joint.stillTollerance:
                    still = False

        if still == False:
            return False
        elif still == True:
            return True

    def circle(self):
        jointTrail = self.jointTrail
        radii = []
        drew_circle = False
        if len(jointTrail) == self.trailLength and self.still() == False:
            drew_circle = True
            __, center = Joint.averagePoint(jointTrail)
            for point in jointTrail:
                vector = [point[0] - center[0], point[1] - center[1]]
                radii.append(Joint.mag(vector))
            ave_radius = np.mean(radii)
            for point in jointTrail:
                vector = [point[0] - center[0], point[1] - center[1]]
                magnitude = Joint.mag(vector)
                if abs(magnitude - ave_radius) > Joint.circleStrayTollerance:
                    drew_circle = False
                if ave_radius < Joint.circleMinRadius:
                    drew_circle = False

            if Joint.mag([jointTrail[0][0] - jointTrail[self.trailLength - 1][0],
                          jointTrail[0][1] - jointTrail[self.trailLength - 1][1]]) > Joint.circleClosedTollerance:
                drew_circle = False

        if drew_circle == False:
            return False, 0, 0
        elif drew_circle == True:
            return True, center, ave_radius

    def hLine(self):
        jointTrail = self.jointTrail
        vel = Joint.deriv(jointTrail)
        length = len(jointTrail)

        drew_hLine = False
        if len(jointTrail) == self.trailLength and self.still() == False:
            drew_hLine = True
            for i in range(0, length - 1):
                if abs(jointTrail[i][1] - jointTrail[i + 1][1]) > Joint.lineStrayTollerance:
                    drew_hLine = False
                if vel[i][0] < Joint.lineMinVelocity:
                    drew_hLine = False
                if i < length - 2:
                    if abs(vel[i][0] - vel[i + 1][0]) > Joint.lineSpacingTollerance:
                        drew_hLine = False
        if drew_hLine == False:
            return False, 0, 0
        elif drew_hLine == True:
            return True, jointTrail[0], jointTrail[len(jointTrail) - 1]

    def vLine(self):
        jointTrail = self.jointTrail
        vel = Joint.deriv(jointTrail)
        length = len(jointTrail)

        drew_vLine = False
        if len(jointTrail) == self.trailLength and self.still() == False:
            drew_vLine = True
            for i in range(0, length - 1):
                if abs(jointTrail[i][0] - jointTrail[i + 1][0]) > Joint.lineStrayTollerance:
                    drew_vLine = False
                if vel[i][1] < Joint.lineMinVelocity:
                    drew_vLine = False
                if i < length - 2:
                    if abs(vel[i][1] - vel[i + 1][1]) > Joint.lineSpacingTollerance:
                        drew_vLine = False
        if drew_vLine == False:
            return False, 0, 0
        elif drew_vLine == True:
            return True, jointTrail[0], jointTrail[len(jointTrail) - 1]

    def gatherPoints(self, img):
        mask = cv2.inRange(img, self.lowerRange, self.upperRange)
        sample = Joint.sample
        white = []
        for row in range(0, Joint.Nrow, sample):
            for col in range(0, Joint.Ncol, sample):
                if mask[row, col] == 255:
                    white.append([col, row])
        self.mask = mask
        self.maskList = white
        return mask, white

    def connectTo(self, joint2):
        if (self.x != 0 or self.y != 0) and (joint2.x != 0 or joint2.y != 0):
            cv2.line(Joint.img, (self.x, self.y), (joint2.x, joint2.y), Joint.BONECOLOR, 2)

    def drawTracking(self, img):
        if not self.jointTrail:
            return
        point1 = self.jointTrail[0]
        if len(self.jointTrail) == 1:
            cv2.line(img, (point1[0], point1[1]), (point1[0] + 2, point1[1] + 2), Joint.DRAW_COLOR, 2)
        elif len(self.jointTrail) > 1:
            for i in range(0, len(self.jointTrail) - 1):
                cv2.line(img, (self.jointTrail[i][0], self.jointTrail[i][1]),
                         (self.jointTrail[i + 1][0], self.jointTrail[i + 1][1]), Joint.DRAW_COLOR, 2)

    def drawJoint(self, img):
        if self.x != 0 or self.y != 0:
            if self.state[0] == "heal":
                cv2.circle(img, (self.x, self.y), 150, Joint.YELLOW, 80)
            if self.state[0] == "shield":
                cv2.circle(img, (self.x, self.y), 200, Joint.RED, 20)
            else:
                cv2.circle(img, (self.x, self.y), 10, Joint.JOINT_COLOR, 2)

    def findJoint_ContourMethod(self):
        if not hasattr(self, 'mask') or self.mask is None:
            return Joint()
        mask = self.mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = []
        for contour in contours:
            contour_sizes.append([cv2.contourArea(contour), contour])
        if contour_sizes != []:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            if len(biggest_contour) > Joint.minContourSize:
                xf, yf, w, h = cv2.boundingRect(biggest_contour)
                return Joint(x=xf + int(w / 2), y=yf + int(h / 2))
            else:
                return Joint()
        else:
            return Joint()

    def findJoint_AverageMethod(self):
        if not hasattr(self, 'maskList') or self.maskList is None:
            return Joint()
        maskList = self.maskList
        count, ave = Joint.averagePoint(maskList)
        joint = Joint(x=ave[0], y=ave[1], lowerRange=self.lowerRange, upperRange=self.upperRange)
        return joint

    def findJoint_HaarMethod(self, gray):
        type = self.type

        if type == 'head':
            haar = Joint.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        elif type == 'fist':
            haar = Joint.fist_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        elif type == 'upper':
            haar = Joint.upper_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        elif type == 'smile':
            haar = Joint.smile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        else:
            print("joint type not recognized")
            return Joint()

        if len(haar) == 0:
            print(f"No {type} detected")
        else:
            print(f"{type} detected: {haar}")

        joint_found = False
        for (xf, yf, w, h) in haar:
            joint_found = True
            return Joint(x=xf + int(w / 2), y=yf + int(h / 2))

        if joint_found == False:
            if type == 'head':
                haar = Joint.face_glasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (xf, yf, w, h) in haar:
                joint_found = True
                return Joint(x=xf + int(w / 2), y=yf + int(h / 2))

        if joint_found == False:
            return Joint()

    @staticmethod
    def averagePoint(list_):
        totalx = 0
        totaly = 0
        count = 0

        for item in list_:
            totalx += item[0]
            totaly += item[1]
            count += 1
        if count > 0:
            return count, [int(totalx / count), int(totaly / count)]
        else:
            return count, [0, 0]

    @staticmethod
    def deriv(list):
        vel = []
        for i in range(0, len(list)):
            vel.append("zero")
        if len(list) > 1:
            for i in range(0, len(list) - 1):
                vel[i] = [list[i + 1][0] - list[i][0], list[i + 1][1] - list[i][1]]
        vel = [v for v in vel if v != "zero"]
        return vel

    @staticmethod
    def mag(vector):
        x = vector[0]
        y = vector[1]
        return (x ** 2 + y ** 2) ** .5

    def updateJoint(self, img, hsv):
        self.state[1] += 1

        if self.type == 'color':
            self.mask, self.maskList = self.gatherPoints(hsv)
            joint_found = self.findJoint_ContourMethod()
            self.x = joint_found.x
            self.y = joint_found.y

            if self.track == 1:
                self.storeTrack()
            if self.draw == 1:
                self.drawJoint(img)
            if self.drawTrack == 1:
                self.drawTracking(img)

        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            joint_found = self.findJoint_HaarMethod(gray)
            self.x = joint_found.x
            self.y = joint_found.y

            if self.track == 1:
                self.storeTrack()
        if self.draw == 1:
            self.drawJoint(img)
        if self.drawTrack == 1:
            self.drawTracking(img)

'''------------------------------------------GLOBAL VARS---------------------------------'''
LRangeBlk = np.array([0, 0, 0])
URangeBlk = np.array([2, 2, 2])
LRangeBlu = np.array([75, 0, 0])
URangeBlu = np.array([255, 150, 100])
LRangeOra = np.array([0, 0, 150])
URangeOra = np.array([200, 200, 255])
LRangeHSVBlu = np.array([100, 0, 0])
URangeHSVBlu = np.array([120, 255, 100])

'''------------------------------------------INIT----------------------------------------'''
cap = cv2.VideoCapture(0)
Joint.check_cascades()

jointHead = Joint(type='head')
jointFist = Joint(type='fist')
jointUpper = Joint(type='upper')
# You can add more as needed

Joint.totalJointsList.append(jointHead)
Joint.totalJointsList.append(jointFist)
Joint.totalJointsList.append(jointUpper)

circles = []
lines = []

'''------------------------------------------MAIN LOOP-----------------------------------'''
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    Joint.img = cv2.resize(frame, (Joint.Ncol, Joint.Nrow))
    Joint.hsv = cv2.cvtColor(Joint.img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(Joint.img, cv2.COLOR_BGR2GRAY)

    # --- Face and eye detection and drawing ---
    faces = Joint.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        # Draw rectangle around the whole face
        cv2.rectangle(Joint.img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(Joint.img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = Joint.img[y:y + h, x:x + w]
        eyes = Joint.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- Your existing joint update logic (for other joints, e.g. fist, upper) ---
    for joint in Joint.totalJointsList:
        joint.updateJoint(Joint.img, Joint.hsv)

    cv2.imshow('Joint.img', Joint.img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()