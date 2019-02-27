import os
import argparse

import numpy as np
import torch
import cv2

from goturn import TrackerGOTURN

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man', type=str,
                    help='path to video frames')
parser.add_argument('-s', '--save-directory',
                    default='../result',
                    type=str, help='path to save directory')

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
PADDING = 2

def axis_aligned_iou(boxA, boxB):
    # make sure that x1,y1,x2,y2 of a box are valid
    assert(boxA[0] <= boxA[2])
    assert(boxA[1] <= boxA[3])
    assert(boxB[0] <= boxB[2])
    assert(boxB[1] <= boxB[3])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])


def save(im, bb, gt_bb, idx):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    bb = [int(val) for val in bb]  # GOTURN output
    gt_bb = [int(val) for val in gt_bb]  # groundtruth box
    # plot GOTURN predictions with red rectangle
    im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                       (0, 0, 255), 2)
    # plot annotations with white rectangle
    im = cv2.rectangle(im, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]),
                       (255, 255, 255), 2)
    save_path = os.path.join(args.save_directory, str(idx)+'.jpg')
    cv2.imwrite(save_path, im)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('img', im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def show_webcam(arg, mirror=False):
    global tracker, initialize
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0)
    frameNum = 0
    outputDir = None
    outputBoxToDraw = None
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        origImg = img.copy()
        if mousedown:
            cv2.rectangle(img,
                    (int(boxToDraw[0]), int(boxToDraw[1])),
                    (int(boxToDraw[2]), int(boxToDraw[3])),
                    [0,0,255], PADDING)
        elif mouseupdown:
            if initialize:
                boxToDraw[2] = boxToDraw[2] - boxToDraw[0]
                boxToDraw[3] = boxToDraw[3] - boxToDraw[1]
                tracker.init(img, boxToDraw)
                outputBoxToDraw = boxToDraw
                #outputBoxToDraw = tracker.track('webcam', img[:,:,::-1], boxToDraw)
                initialize = False
            else:
                outputBoxToDraw = tracker.update(img)
                #outputBoxToDraw = tracker.track('webcam', img[:,:,::-1])
            cv2.rectangle(img,
                    (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                    (int(outputBoxToDraw[0]+outputBoxToDraw[2]), int(outputBoxToDraw[1]+outputBoxToDraw[3])),
                    [0,0,255], PADDING)
        cv2.imshow('Webcam', img)
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        frameNum += 1
    cv2.destroyAllWindows()

def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(args.data_directory,
                    args.model_weights,
                    device)
    
    # save initial frame with bounding box
    save(tester.img[0][0], tester.prev_rect, tester.prev_rect, 1)
    tester.model.eval()

    # loop through sequence images
    for i in range(tester.len):
        # get torch input tensor
        sample = tester[i]

        # predict box
        bb = tester.get_rect(sample)
        gt_bb = tester.gt[i]
        tester.prev_rect = bb
        # save current image with predicted rectangle and gt box
        im = tester.img[i][1]
        save(im, bb, gt_bb, i+2)

        # print stats
        print('frame: %d, IoU = %f' % (
            i+2, axis_aligned_iou(gt_bb, bb)))


if __name__ == "__main__":
    args = parser.parse_args()
    tracker = TrackerGOTURN(net_path=args.model_weights)
    #main(args)
    show_webcam(args, mirror=True)
