import cv2
#####################################################################

flag_processing = True
video_name = 'videos/test.mp4'
#####################################################################

# define video capture object

cap = cv2.VideoCapture()

(major, minor, _) = cv2.__version__.split(".")
if (major == '3'):
    cv2.ocl.setUseOpenCL(False)

# define display window name

windowName = "Video Input"
windowNameFGP = "Foreground Probabiity"

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if cap.open(video_name):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

    while (flag_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                flag_processing = False
                continue

        # add current frame to background model and retrieve current foreground objects

        fgmask = mog.apply(frame)

        # threshold this and clean it up using dilation with a elliptical mask

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)

        # get current background image (representative of current GMM model)

        bgmodel = mog.getBackgroundImage()

        # display images - input, background and original

        cv2.imshow(windowName,frame)
        cv2.imshow(windowNameFGP,fgmask)

        key = cv2.waitKey(10) & 0xFF

        # if user presses "q" then exit
        if (key == ord('q')):
            flag_processing = False

    # close all windows
    cv2.destroyAllWindows()
else:
    print("No video file specified or camera connected.")