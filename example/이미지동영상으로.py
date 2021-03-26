import os
import cv2
from PIL import Image


print(os.getcwd())

where = "G:\selected pic\새 폴더"
##########################################SETTING PATH

os.chdir(where)
path = where

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir('.'))


for file in os.listdir('.'):
    im = Image.open(os.path.join(path, file))
    width, height = im.size
    mean_width += width
    mean_height += height

mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)


for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png") or file.endswith("bmp"):

        im = Image.open(os.path.join(path, file))


        width, height = im.size
        print(width, height)


        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
        imResize.save(file, 'JPEG', quality=95)
        print(im.filename.split('\\')[-1], " is resized")




def generate_video():
    image_folder = '.'
    video_name = 'contour.avi'
    os.chdir(where)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png") or
              img.endswith("bmp")]


    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 3, (width, height))
    ##################################### FPSSETTING


    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))











    cv2.destroyAllWindows()
    video.release()


generate_video()

#%%
import cv2
import time
import os

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":

    input_loc = r'E:\experiment data\aluminum\2021-03-25\1_Mobile/Trim.mp4'
    output_loc =  r'E:\experiment data\aluminum\2021-03-25\1_Mobile/img/'
    video_to_frames(input_loc, output_loc)