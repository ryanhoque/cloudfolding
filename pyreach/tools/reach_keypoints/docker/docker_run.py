#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "nvidia-docker run -it -v %s:/host \
                                 -v %s:/host/data priya-keypoints" % (os.path.join(os.getcwd(), '..'), '/raid/priya/data/keypoints/datasets')
    #cmd = "nvidia-docker run -it priya-keypoints" 
    #cmd = "docker run --runtime=nvidia -it -v %s:/host priya-keypoints" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
