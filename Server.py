############################################################################################
#
# The MIT License (MIT)
# 
# TASS Facenet Classifier Server
# Copyright (C) 2018 Adam Milton-Barker (AdamMiltonBarker.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Title:         TASS Facenet Classifier Server
# Description:   Serves an API for classification of facial recognition images.
# Configuration: required/confs.json
# Last Modified: 2018-08-09
#
# Example Usage:
#
#   $ python3.5 Server.py
#
############################################################################################

print("")
print("!! Welcome to TASS Facenet Classifier Server, please wait while the program initiates !!")
print("")

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("-- Running on Python "+sys.version)

import time, csv, getopt, json, time, jsonpickle, cv2
import numpy as np

from tools.Helpers import Helpers
from tools.OpenCV import OpenCVHelpers as OpenCVHelpers
from tools.Facenet import FacenetHelpers

from datetime import datetime
from flask import Flask, request, Response
from mvnc import mvncapi as mvnc
from skimage.transform import resize

print("-- Imported Required Modules")

print("-- API Initiating ")
app = Flask(__name__)
print("-- API Intiated ")

class Server():

    def __init__(self):

        self._configs = {}
        self.movidius = None
        self.cameraStream = None
        self.imagePath = None

        self.mean = 128
        self.std = 1/128

        self.categories = []
        self.fgraphfile = None
        self.fgraph = None
        self.reqsize = None

        self.Helpers = Helpers()
        self._configs = self.Helpers.loadConfigs()

        print("-- Server Initiated")

    def CheckDevices(self):

        #mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('!! WARNING! No Movidius Devices Found !!')
            quit()

        self.movidius = mvnc.Device(devices[0])
        self.movidius.OpenDevice()

        print("-- Movidius Connected")

    def allocateGraph(self, graphfile, graphID):
        
        self.fgraph = self.movidius.AllocateGraph(graphfile)

    def loadRequirements(self, graphID):

        with open(self._configs["ClassifierSettings"]["NetworkPath"] + self._configs["ClassifierSettings"]["Graph"], mode='rb') as f:

            self.fgraphfile = f.read()

        self.allocateGraph(self.fgraphfile,"TASS")
        print("-- Allocated TASS Graph OK")

Server = Server()
FacenetHelpers = FacenetHelpers()
Server.CheckDevices()
Server.loadRequirements("TASS")

@app.route('/api/TASS/infer', methods=['POST'])
def TASSinference():

    humanStart = datetime.now()
    clockStart = time.time()

    print("-- FACENET LIVE INFERENCE STARTED: ", humanStart)

    r = request
    nparr = np.fromstring(r.data, np.uint8)

    print("-- Loading Face")
    fileName = "data/captured/TASS/"+str(clockStart)+'.png'
    print("-- Loading Face")
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(fileName,img)
    img = cv2.imread(fileName)
    print("-- Loaded Sample")

    validDir    = Server._configs["ClassifierSettings"]["NetworkPath"] + Server._configs["ClassifierSettings"]["ValidPath"]
    testingDir  = Server._configs["ClassifierSettings"]["NetworkPath"] + Server._configs["ClassifierSettings"]["TestingPath"]

    files = 0
    identified = 0

    test_output = FacenetHelpers.infer(img, Server.fgraph)
    files = files + 1

    for valid in os.listdir(validDir):

        if valid.endswith('.jpg') or valid.endswith('.jpeg') or valid.endswith('.png') or valid.endswith('.gif'):

            valid_output = FacenetHelpers.infer(cv2.imread(validDir+valid), Server.fgraph)
            known, confidence = FacenetHelpers.match(valid_output, test_output)
            if (known=="True"):
                identified = identified + 1
                print("-- MATCH "+valid)
                break

    humanEnd = datetime.now()
    clockEnd = time.time()

    print("")
    print("-- FACENET LIVE INFERENCE ENDED: ", humanEnd)
    print("-- TESTED: ", 1)
    print("-- IDENTIFIED: ", identified)
    print("-- TIME(secs): {0}".format(clockEnd - clockStart))
    print("")

    if identified:

        validPerson = os.path.splitext(valid)[0]
        message = validPerson + " Detected With Confidence " + str(confidence)
        person = validPerson

    else:

        message = "Intruder Detected With Confidence " + str(confidence)
        person = "Intruder"

    response = {
        'Response': 'OK',
        'Results': identified,
        'Person': person,
        'Confidence': str(confidence),
        'ResponseMessage': message
    }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
        
    app.run(host=Server._configs["Cameras"][0]["Stream"], port=Server._configs["Cameras"][0]["StreamPort"])
