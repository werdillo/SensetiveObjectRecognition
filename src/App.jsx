import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import { client } from "./utils/pocketbase";
import labels from "./utils/labels.json";
import "./style/App.css";
import ModelBenchmark from './components/ModelBenchmark';

const App = () => {
  const [loading, setLoading] = useState({ loading: false, progress: 0 }); // изначально не загружаем
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
  const [activeTab, setActiveTab] = useState("benchmark"); // активная вкладка
  const [modelLoaded, setModelLoaded] = useState(false); // загружена ли модель
  const [warmupCompleted, setWarmupCompleted] = useState(false); // завершен ли warmup
  const [warmupProgress, setWarmupProgress] = useState(0); // прогресс warmup

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Легкий warmup только для TensorFlow.js без загрузки моделей
  useEffect(() => {
    const performWarmup = async () => {
      try {
        setWarmupProgress(20);
        
        // Ждем готовности TensorFlow.js
        await tf.ready();
        setWarmupProgress(50);
        
        // Создаем простые операции для инициализации WebGL backend
        const warmupTensor = tf.zeros([1, 224, 224, 3]);
        const result = tf.add(warmupTensor, tf.scalar(1));
        await result.data(); // Заставляем выполниться на GPU
        warmupTensor.dispose();
        result.dispose();
        setWarmupProgress(80);
        
        // Создаем еще несколько операций для стабилизации
        const testTensor = tf.randomNormal([1, 100, 100, 3]);
        const convTest = tf.conv2d(testTensor, tf.randomNormal([3, 3, 3, 16]), 1, 'same');
        await convTest.data();
        testTensor.dispose();
        convTest.dispose();
        
        setWarmupProgress(100);
        
        setTimeout(() => {
          setWarmupCompleted(true);
        }, 300);
        
      } catch (error) {
        console.error('Warmup failed:', error);
        setWarmupCompleted(true);
      }
    };

    performWarmup();
  }, []); 

  const loadModel = async () => {
    if (modelLoaded && model.net) return; 
    
    setLoading({ loading: true, progress: 0 });
    const startLoadTime = performance.now();

    const yoloModel = await tf.loadGraphModel(
      `${window.location.href}/${selectedModel}_web_model/model.json`,
      {
        onProgress: (fractions) => {
          setLoading({ loading: true, progress: fractions });
        },
      }
    );

    const endLoadTime = performance.now();
    const loadDuration = endLoadTime - startLoadTime;

    // warming up model
    const dummyInput = tf.ones(yoloModel.inputs[0].shape);
    const warmupResults = yoloModel.execute(dummyInput);

    // Правильная очистка warmup результатов
    if (Array.isArray(warmupResults)) {
      warmupResults.forEach(tensor => tensor.dispose());
    } else {
      warmupResults.dispose();
    }
    dummyInput.dispose();

    setLoading({ loading: false, progress: 1 });
    setModel({
      net: yoloModel,
      inputShape: yoloModel.inputs[0].shape,
    });
    setModelLoaded(true);
    
    if (initialLoadTime === null) {
      setInitialLoadTime(loadDuration);
      const dataInitial = {
        "device": device,
        "timeMs": loadDuration.toFixed(2)
      }
      client.collection('initialLoad').create(dataInitial);
    } else {
      setLoadTime(loadDuration);
      const data = {
        "device": device,
        "model": selectedModel,
        "time": loadDuration.toFixed(2)
      }
      client.collection('modelLoad').create(data);
    }
  };

  // Загрузка модели при смене модели (только если находимся на вкладке Detection)
  useEffect(() => {
    if (activeTab === "detection" && device && warmupCompleted) {
      tf.ready().then(loadModel);
    }
  }, [selectedModel, activeTab, device, warmupCompleted]);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    setModelLoaded(false); // сбросить состояние загрузки при смене модели
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
      setDevice(deviceInput.trim());
    } else {
      alert("Please enter a valid device name.");
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    // Загружать модель только при переходе на вкладку Detection
    if (tab === "detection" && device && !modelLoaded && warmupCompleted) {
      tf.ready().then(loadModel);
    }
  };

  const models = ["yolo11n", "yolo11s", "yolo11m"];

  // Показываем экран warmup если он еще не завершен
  if (!warmupCompleted) {
    return (
      <div className="App">
        <Loader>
          <div style={{ textAlign: 'center' }}>
            <b>🚀 Initializing TensorFlow.js...</b>
            <div style={{ 
              marginTop: '20px',
              width: '300px',
              backgroundColor: '#e0e0e0',
              borderRadius: '10px',
              overflow: 'hidden',
              margin: '20px auto'
            }}>
              <div 
                style={{ 
                  width: `${warmupProgress}%`, 
                  backgroundColor: '#007bff', 
                  height: '20px', 
                  borderRadius: '10px',
                  transition: 'width 0.3s',
                  background: 'linear-gradient(90deg, #007bff 0%, #0056b3 100%)'
                }}
              />
            </div>
            <div style={{ fontSize: '14px', color: '#666' }}>
              {warmupProgress.toFixed(0)}% - Preparing WebGL backend...
            </div>
          </div>
        </Loader>
      </div>
    );
  }

  return (
    <div className="App">
      {loading.loading && <Loader> <b>Loading model... {(loading.progress * 100).toFixed(2)}%</b></Loader>}
      <div className="card">
        <div className="card-content">

          {device === null ? (
            <>
              <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                <div style={{ 
                  display: 'inline-block',
                  padding: '10px 20px',
                  backgroundColor: '#d4edda',
                  border: '1px solid #c3e6cb',
                  borderRadius: '6px',
                  color: '#155724',
                  fontSize: '14px'
                }}>
                  ✅ TensorFlow.js initialized and ready!
                </div>
              </div>
              <b>Please enter your device model</b>
              <input onChange={e => setDeviceInput(e.target.value)}/>
              <button onClick={handleDeviceSave}>Submit</button>
            </>
          ) : (
            <>
              <h3>Your device: {device}</h3>
              
              {/* Навигация по вкладкам */}
              <div className="tabs-navigation">
                <button 
                  onClick={() => handleTabChange("benchmark")}
                  className={activeTab === "benchmark" ? "tab-active" : "tab-inactive"}
                >
                  Benchmark
                </button>
                <button 
                  onClick={() => handleTabChange("detection")}
                  className={activeTab === "detection" ? "tab-active" : "tab-inactive"}
                >
                  Detection
                </button>
              </div>

              {activeTab === "benchmark" ? (
                <ModelBenchmark 
                  device={device}
                  client={client}
                  labels={labels}
                />
              ) : (
                <>
                  <div className="buttons-wrapper">
                    {models.map((modelName) => (
                      <button
                        key={modelName}
                        onClick={() => handleModelChange({ target: { value: modelName } })}
                        disabled={loading.loading}
                        style={{
                          backgroundColor: selectedModel === modelName ? "#007bff" : "#f8f9fa",
                          color: selectedModel === modelName ? "#fff" : "#000",
                        }}
                      >
                        {modelName.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  <div className="content-container">
                    <div className="content">
                      <img
                        src="#"
                        ref={imageRef}
                        onLoad={handleImageLoad}
                      />
                      <video
                        autoPlay
                        muted
                        ref={cameraRef}
                        onPlay={() => handleVideoPlay(cameraRef.current)}
                      />
                      <video
                        autoPlay
                        muted
                        ref={videoRef}
                        onPlay={() => handleVideoPlay(videoRef.current)}
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
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;