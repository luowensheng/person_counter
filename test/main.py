
from opencv_stream import VideoStreamer, FpsDrawer
from person_counter.model import PersonCounterModel, PersonCounterOutput
import numpy as np
import os

VIDEO_DIR = "D:/project/facebodydetection/facebodydetect/app/src/videos"
def get_video():
   paths = [ os.path.join(VIDEO_DIR, p) for p in os.listdir(VIDEO_DIR)]
   return np.random.choice(paths)



stream = VideoStreamer.from_video_input(get_video())
fps = FpsDrawer()

model = PersonCounterModel(yolo_weights="D:/project/person_tracker/person_tracking/detector.pt")


@stream.on_next_frame()
def index(frame: np.ndarray):
   
   result = model.predict(frame) 

   if result.is_ok():
      output: PersonCounterOutput = result.unwrap()
      output.draw(frame)
   else:
    raise result.exception

   fps.draw(frame)


stream.start()



"""


from opencv_stream import VideoStreamer, FpsDrawer
from person_counter.model import PersonCounterModel, PersonCounterOutput
import numpy as np
import os

VIDEO_DIR = input("Enter directory with video files: ")
def get_video():
   paths = [ os.path.join(VIDEO_DIR, p) for p in os.listdir(VIDEO_DIR) if p.endswith(".mp4")]
   return np.random.choice(paths)



stream = VideoStreamer.from_video_input(get_video())
fps = FpsDrawer()

model = PersonCounterModel()


@stream.on_next_frame()
def index(frame: np.ndarray):
   
   result = model.predict(frame) 

   if result.is_ok():
      output: PersonCounterOutput = result.unwrap()
      output.draw(frame)
   else:
    raise result.exception

   fps.draw(frame)


stream.start()


"""