import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import { client } from "./utils/pocketbase";
import labels from "./utils/labels.json";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape
  const [detectionTime, setDetectionTime] = useState(0); // state for storing detection time
  const [loadTime, setLoadTime] = useState(0); // время загрузки модели
  const [initialLoadTime, setInitialLoadTime] = useState(null); // время первой загрузки
  const [selectedModel, setSelectedModel] = useState("yolo11n"); // выбранная модель
  const [device, setDevice] = useState(null); // выбранная модель
  const [deviceInput, setDeviceInput] = useState("");

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      setLoading({ loading: true, progress: 0 }); // сброс состояния загрузки
      const startLoadTime = performance.now(); // начало замера времени загрузки

      const yoloModel = await tf.loadGraphModel(
        `${window.location.href}/${selectedModel}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      );

      const endLoadTime = performance.now(); // конец замера времени загрузки
      const loadDuration = endLoadTime - startLoadTime;


      // warming up model
      const dummyInput = tf.ones(yoloModel.inputs[0].shape);
      const warmupResults = yoloModel.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yoloModel,
        inputShape: yoloModel.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
      
      if (initialLoadTime === null) {
        setInitialLoadTime(loadDuration); // сохранить время первой загрузки
        const dataInitial = {
          "device": device,
          "timeMs": loadDuration.toFixed(2)
        }
        client.collection('initialLoad').create(dataInitial);
      } else {
        setLoadTime(loadDuration); // сохранить время последующей загрузк
        const data = {
          "device": device,
          "model": selectedModel,
          "time": loadDuration.toFixed(2)
        }
        client.collection('modelLoad').create(data);
      }
    };

    tf.ready().then(loadModel);
  }, [selectedModel]); // перезагрузка модели при изменении выбранной модели

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value); // изменение модели
  };

  const handleImageLoad = async () => {
    const startTime = performance.now();
    const res = await detect(imageRef.current, model, canvasRef.current);
    const endTime = performance.now();
    const resTime = (endTime - startTime)
    setDetectionTime(resTime);

    const safeValue = (value) => (value === undefined || value === null ? "-" : value);

    const data = {
      device: safeValue(device),
      class: safeValue(labels?.[res.classes[0]]),
      precission: safeValue(res?.scores?.[0]?.toFixed(2)),
      timeMs: safeValue(resTime),
      model: safeValue(selectedModel),
    };
    await client.collection('imageRecognition').create(data);
  };

  const handleVideoPlay = async (video) => {
    const startTime = performance.now();
    await detectVideo(video, model, canvasRef.current);
    const endTime = performance.now();
    setDetectionTime(endTime - startTime);
  };

  const handleDeviceSave = () => {
    if (deviceInput.trim() !== "") {
      setDevice(deviceInput.trim()); // сохранить имя устройства
    } else {
      alert("Please enter a valid device name.");
    }
  };
  const models = ["yolo11n", "yolo11s", "yolo11m"];
  return (
    <div className="App">
        {loading.loading && <Loader> <b>Loading model... {(loading.progress * 100).toFixed(2)}%</b></Loader>}
      <div className="card">
      <div className="card-content">

        {device === null ?
          <>
            Please enter your device model
            <input onChange={e => setDeviceInput(e.target.value)}/>
            <button onClick={handleDeviceSave}>Save</button>
          </>
        :
        <>
        <h3>Your device {device}</h3>
        <div className="buttons-wrapper">
          {models.map((model) => (
            <button
              key={model}
              onClick={() => handleModelChange({ target: { value: model } })}
              disabled={loading.loading} // Блокировка во время загрузки
              style={{
                backgroundColor: selectedModel === model ? "#007bff" : "#f8f9fa", // Синяя кнопка для выбранной модели
                color: selectedModel === model ? "#fff" : "#000", // Белый текст для выбранной модели
              }}
            >
              {model.toUpperCase()} {/* Отображаем имя модели */}
            </button>
          ))}
        </div>


        <div className="content-container ">
          <div className="content">
            <img
              src="#"
              ref={imageRef}
              onLoad={handleImageLoad} // вызов функции обработки изображения
            />
            <video
              autoPlay
              muted
              ref={cameraRef}
              onPlay={() => handleVideoPlay(cameraRef.current)} // вызов функции обработки видео
            />
            <video
              autoPlay
              muted
              ref={videoRef}
              onPlay={() => handleVideoPlay(videoRef.current)} // вызов функции обработки видео
            />
            <canvas width={model.inputShape[1]} height={model.inputShape[2]} ref={canvasRef} />
          </div>
        </div>

        <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
          <div className="info-panel">
            <div className="info-item">
              <span className="info-label">Detection Time:</span>
              <span className="info-value">{detectionTime.toFixed(2)} ms</span>
            </div>
            <div className="info-item">
              <span className="info-label">Model Load Time:</span>
              <span className="info-value">{loadTime.toFixed(2)} ms</span>
            </div>
            <div className="info-item">
                <span className="info-label">Initial Time:</span>
                <span className="info-value">{ initialLoadTime !== null ? initialLoadTime.toFixed(2) : "0.00"} ms</span>
            </div>
          </div>
          </>
        }
      </div>
    </div>
  </div>
  );
};

export default App;