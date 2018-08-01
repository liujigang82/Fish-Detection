import cv2
import sys

#####################################################################

keep_processing = True
video_name = 'videos/test.mp4'
#####################################################################

# define video capture object

cap = cv2.VideoCapture();

(major, minor, _) = cv2.__version__.split(".")
if (major == '3'):
    cv2.ocl.setUseOpenCL(False);

# define display window name

windowName = "Live Camera Input"; # window name
windowNameBG = "Background Model"; # window name
windowNameFG = "Foreground Objects"; # window name
windowNameFGP = "Foreground Probabiity"; # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((len(sys.argv) == 2) and (cap.open(str(sys.argv[1]))))
    or (cap.open('videos/test.mp4'))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL);

    # create GMM background subtraction object (using default parameters - see manual)

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read();
            #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False;
                continue;

        # add current frame to background model and retrieve current foreground objects

        fgmask = mog.apply(frame);
        fgmask = cv2.blur(fgmask, (5, 5))

        #fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        #fgmask = cv2.bilateralFilter(fgmask, 9, 75, 75)
        # threshold this and clean it up using dilation with a elliptical mask

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1];
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3);

        # get current background image (representative of current GMM model)

        bgmodel = mog.getBackgroundImage();

        # display images - input, background and original

        cv2.imshow(windowName,frame);
        cv2.imshow(windowNameFG,fgdilated);
        cv2.imshow(windowNameFGP,fgmask);
        cv2.imshow(windowNameBG, bgmodel);

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(10) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False;

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");