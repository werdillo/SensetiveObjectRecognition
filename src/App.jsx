import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape
  const [detectionTime, setDetectionTime] = useState(0); // state for storing detection time
  const [loadTime, setLoadTime] = useState(0); // время загрузки модели
  const [selectedModel, setSelectedModel] = useState("yolo11s"); // выбранная модель

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
      setLoadTime(endLoadTime - startLoadTime); // обновляем состояние времени загрузки

      // warming up model
      const dummyInput = tf.ones(yoloModel.inputs[0].shape);
      const warmupResults = yoloModel.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yoloModel,
        inputShape: yoloModel.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    };

    tf.ready().then(loadModel);
  }, [selectedModel]); // перезагрузка модели при изменении выбранной модели

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value); // изменение модели
  };

  const handleImageLoad = async () => {
    const startTime = performance.now();
    await detect(imageRef.current, model, canvasRef.current);
    const endTime = performance.now();
    setDetectionTime(endTime - startTime);
  };

  const handleVideoPlay = async (video) => {
    const startTime = performance.now();
    await detectVideo(video, model, canvasRef.current);
    const endTime = performance.now();
    setDetectionTime(endTime - startTime);
  };
  const models = ["yolo11n", "yolo11s", "yolo11m"];
  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      {/* Тумблер выбора модели */}
      <label htmlFor="model-selector">Select Model:</label>
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

      {loadTime > 0 && (
        <p>
          Model load time: <strong>{loadTime.toFixed(2)} ms</strong>
        </p>
      )}

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

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />

      {detectionTime > 0 && (
        <div className="detection-time">
          <p>Time taken for detection: {detectionTime.toFixed(2)} ms</p>
        </div>
      )}
    </div>
  );
};

export default App;