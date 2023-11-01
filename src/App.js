import { useRef, useEffect, useState } from "react";

import * as handpose from "@tensorflow-models/handpose";
import { create } from "@tensorflow-models/knn-classifier";
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; 

import Webcam from "react-webcam";

import './App.css';

import { drawHand } from "./utils";

const App = () => {
  const [classifier, setClassifier] = useState(null);
  const [model, setModel] = useState(null);
  const [hasExamples, setHasExamples] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const runHandpose = async () => {
    const net = await handpose.load();

    setInterval(() => {
      detect(net);
      makePredictions();

    }, 200);
  };

  const detect = async (net) => {
    const currentWebcam = webcamRef.current;

    if (!currentWebcam) {
      return;
    }

    const video = currentWebcam.video;

    if (video.readyState !== 4) {
      return;
    }

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const hand = await net.estimateHands(video);
    const ctx = canvasRef.current.getContext("2d");
    drawHand(hand, ctx);
  };

  const addExample = async (classId) => {
    const currentWebcam = webcamRef.current;

    if (!currentWebcam) {
      return;
    }

    const video = currentWebcam.video;

    if (video.readyState !== 4) {
      return;
    }

    const myModel = await mobilenet.load();

    setModel(myModel);
    await tf.ready();
    const activation = myModel.infer(video, true);

    classifier.addExample(activation, classId);

    setHasExamples(true);
  };

  const makePredictions = async () => {
    if (!hasExamples) {
      return;
    }

    const currentWebcam = webcamRef.current;

    if (!currentWebcam) {
      return;
    }

    const video = currentWebcam.video;

    if (video.readyState !== 4) {
      return;
    }

    if (!model) {
      return;
    }
  
    
      

    if (classifier.getNumClasses() > 0.66) {
      const activation = model.infer(video, 'conv_preds');
      const result = await classifier.predictClass(activation);
      
          const classes = ['A', 'B', 'C'];
          document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      console.log(classes[result.label])
            
      }
    
          
  };

  const runClassifier = () => {
    setClassifier(create());
  };

  useEffect(() => {
    runHandpose();
    runClassifier();
    tf.setBackend("webgl");
  }, [webcamRef, canvasRef, hasExamples]);

  return (
    <div className="App">
      <header className="App-header">
        <div id="console"></div>
        <Webcam ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480
          }} 
          screenshotFormat="video/mp4,video/x-m4v,video/*"
          />
        <canvas ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480
          }} />
      </header>
      <button
        onClick={() => addExample(0)}
        style={{
          marginTop: "50px"
        }}
      >
        A
      </button>
    </div>
  );
};

export default App;
