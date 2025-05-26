import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { detect } from '../utils/detect';
import { client } from '../utils/pocketbase';

const ModelBenchmark = ({ device, client, labels }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState('');
  const [results, setResults] = useState([]);
  const [isMobile, setIsMobile] = useState(false);
  const canvasRef = useRef(null);

  // Определяем мобильное устройство
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const models = ['yolo11n', 'yolo11s', 'yolo11m'];
  
  // Предопределенный список тестовых изображений
  const imageFiles = [
    'card1.png',
    'card2.jpeg',
    'card3.png',
    'card4.jpg',
    'card5.jpg',
    'card6.jpg',
    'card7.jpg',
    'card8.jpg',
    'card9.jpg',
    'card10.jpg',
    'id1.jpg',
    'id2.jpg',
    'id3.jpg',
    'id4.jpg',
    'id5.jpg',
    'id6.jpg',
    'id7.jpg',
    'id8.jpg',
    'id9.jpg',
    'id10.jpg',
    'face1.jpg',
    'face2.jpg',
    'face3.jpg',
    'face4.jpg',
    'face5.jpg',
    'face6.jpg',
    'face7.jpg',
    'face8.jpg',
    'face9.jpg',
    'face10.jpg',
    'signature1.jpg',
    'signature2.jpg',
    'signature3.jpg',
    'signature4.jpg',
    'signature5.jpg',
    'signature6.jpg',
    'signature7.jpg',
    'signature8.jpg',
    'signature9.jpg',
    'signature10.jpg',
  ];

  const loadModel = async (modelName) => {
    try {
      setCurrentStatus(`Loading ${modelName}...`);
      
      // Проверяем доступную память перед загрузкой больших моделей
      const memBefore = tf.memory();
      if (modelName === 'yolo11m' && memBefore.numTensors > 10) {
        setCurrentStatus(`Preparing memory for ${modelName}...`);
        await forceGarbageCollection();
      }
      
      // НАЧИНАЕМ замер времени только здесь
      const startTime = performance.now();
      
      const model = await tf.loadGraphModel(
        `/${modelName}_web_model/model.json`,
        {
          fetchOptions: {
            cache: 'no-cache' // Отключаем кэш для честного тестирования
          }
        }
      );
      
      setCurrentStatus(`Warming up ${modelName}...`);
      
      // Осторожный warming up для больших моделей на iOS
      const dummyInput = tf.ones(model.inputs[0].shape);
      const warmupResults = model.execute(dummyInput);
      
      // Правильная очистка тензоров
      if (Array.isArray(warmupResults)) {
        warmupResults.forEach(tensor => {
          if (tensor && typeof tensor.dispose === 'function') {
            tensor.dispose();
          }
        });
      } else if (warmupResults && typeof warmupResults.dispose === 'function') {
        warmupResults.dispose();
      }
      
      if (dummyInput && typeof dummyInput.dispose === 'function') {
        dummyInput.dispose();
      }
      
      // ЗАКАНЧИВАЕМ замер времени здесь (до очистки памяти)
      const loadTime = performance.now() - startTime;
      
      return {
        net: model,
        inputShape: model.inputs[0].shape,
        loadTime: loadTime
      };
    } catch (error) {
      console.error(`Failed to load model ${modelName}:`, error);
      
      // Специальная обработка для больших моделей на мобильных устройствах
      if (modelName === 'yolo11m' && (navigator.userAgent.includes('iPhone') || navigator.userAgent.includes('iPad'))) {
        throw new Error(`Model ${modelName} requires too much memory for this device`);
      }
      
      throw error;
    }
  };

  const detectImage = async (image, model, canvasRef) => {
    const startTime = performance.now();
    
    try {
      // Используем вашу функцию detect
      const result = await detect(image, model, canvasRef.current);
      
      const detectionTime = performance.now() - startTime;
      
      return {
        time: detectionTime,
        score: result.scores?.[0] || 0,
        class: result.classes?.[0] || -1,
        detections: result.scores?.length || 0
      };
    } catch (error) {
      console.error(`Detection error:`, error);
      return {
        time: 0,
        score: 0,
        class: -1,
        detections: 0
      };
    }
  };

  const loadImage = (src) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(`Failed to load ${src}`);
      img.src = src;
    });
  };

  // Функция принудительной очистки памяти с дополнительными проверками
  const forceGarbageCollection = async () => {
    // Принудительная очистка памяти TensorFlow.js
    await tf.nextFrame();
    await tf.nextFrame(); // Двойная очистка для iOS
    
    // Проверяем использование памяти
    const memInfo = tf.memory();
    console.log(`Memory: ${memInfo.numTensors} tensors, ${(memInfo.numBytes / 1024 / 1024).toFixed(1)} MB`);
    
    // Если много тензоров - ждем еще несколько кадров
    if (memInfo.numTensors > 20) {
      await tf.nextFrame();
      await tf.nextFrame();
      await tf.nextFrame();
    }
    
    // Дополнительная пауза для iOS
    if (navigator.userAgent.includes('iPhone') || navigator.userAgent.includes('iPad')) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  };

  // Функция для сохранения результатов в PocketBase
  const saveBenchmarkSummary = async (benchmarkResults) => {
    try {
      setCurrentStatus('Saving to database...');

      const yolo11nResult = benchmarkResults.find(r => r.model === 'yolo11n');
      const yolo11sResult = benchmarkResults.find(r => r.model === 'yolo11s');
      const yolo11mResult = benchmarkResults.find(r => r.model === 'yolo11m');

      const benchmarkData = {
        device: device,
        yolo11n_load_time: yolo11nResult?.loadTime?.toFixed(2) || "0",
        yolo11n_avg_detection: yolo11nResult?.avgDetectionTime?.toFixed(2) || "0",
        yolo11n_avg_accuracy: yolo11nResult ? (yolo11nResult.avgScore * 100).toFixed(2) : "0",
        yolo11s_load_time: yolo11sResult?.loadTime?.toFixed(2) || "0",
        yolo11s_avg_detection: yolo11sResult?.avgDetectionTime?.toFixed(2) || "0",
        yolo11s_avg_accuracy: yolo11sResult ? (yolo11sResult.avgScore * 100).toFixed(2) : "0",
        yolo11m_load_time: yolo11mResult?.loadTime?.toFixed(2) || "0",
        yolo11m_avg_detection: yolo11mResult?.avgDetectionTime?.toFixed(2) || "0",
        yolo11m_avg_accuracy: yolo11mResult ? (yolo11mResult.avgScore * 100).toFixed(2) : "0",
        fullData: JSON.stringify({
          device: device,
          timestamp: new Date().toISOString(),
          results: benchmarkResults
        })
      };

      await client.collection('Benchmark').create(benchmarkData);
      setCurrentStatus('Benchmark completed and saved!');
      
    } catch (error) {
      console.error('Save error:', error);
      setCurrentStatus('Benchmark completed');
    }
  };

  const runBenchmark = async () => {
    setIsRunning(true);
    setResults([]);
    setProgress(0);
    
    const benchmarkResults = [];
    const totalTests = models.length * imageFiles.length;
    let currentTest = 0;
    
    for (const modelName of models) {
      try {
        // Дополнительная очистка памяти перед каждой моделью
        setCurrentStatus(`Preparing memory for ${modelName}...`);
        await forceGarbageCollection();
        
        // Проверяем память перед загрузкой yolo11m
        const memInfo = tf.memory();
        if (modelName === 'yolo11m' && memInfo.numBytes > 100 * 1024 * 1024) { // Больше 100MB
          setCurrentStatus(`Insufficient memory for ${modelName}, skipping...`);
          benchmarkResults.push({
            model: modelName,
            loadTime: 0,
            avgDetectionTime: 0,
            avgScore: 0,
            totalDetections: 0,
            images: [],
            error: 'Insufficient memory'
          });
          continue;
        }
        
        const model = await loadModel(modelName);
        const modelResults = {
          model: modelName,
          loadTime: model.loadTime,
          images: [],
          totalDetections: 0
        };
        
        for (const imageFile of imageFiles) {
          setCurrentStatus(`Testing ${modelName} on ${imageFile}`);
          
          try {
            const image = await loadImage(`/images/${imageFile}`);
            const result = await detectImage(image, model, canvasRef);
            
            const imageResult = {
              imageName: imageFile,
              detectionTime: result.time,
              score: result.score,
              class: result.class,
              className: labels?.[result.class] || 'unknown',
              detections: result.detections
            };
            
            modelResults.images.push(imageResult);
            modelResults.totalDetections += result.detections;
            
          } catch (imgError) {
            console.error(`Error processing ${imageFile}:`, imgError);
          }
          
          currentTest++;
          setProgress((currentTest / totalTests) * 100);
        }
        
        // Вычисление средних значений
        if (modelResults.images.length > 0) {
          modelResults.avgDetectionTime = modelResults.images.reduce((sum, img) => sum + img.detectionTime, 0) / modelResults.images.length;
          modelResults.avgScore = modelResults.images.reduce((sum, img) => sum + img.score, 0) / modelResults.images.length;
        } else {
          modelResults.avgDetectionTime = 0;
          modelResults.avgScore = 0;
        }
        
        benchmarkResults.push(modelResults);
        
        // ВАЖНО: Правильная очистка модели (НЕ включается в время загрузки следующей модели)
        setCurrentStatus(`Cleaning up ${modelName}...`);
        if (model.net && typeof model.net.dispose === 'function') {
          model.net.dispose();
        }
        
        // Принудительная очистка памяти после каждой модели (особенно важно для iOS)
        await forceGarbageCollection();
        
      } catch (error) {
        console.error(`Error with model ${modelName}:`, error);
        
        // Специальная обработка для ошибок памяти на iOS
        const isMemoryError = error.message.includes('memory') || error.message.includes('Memory') || 
                            error.message.includes('allocation') || error.name === 'RangeError';
        
        benchmarkResults.push({
          model: modelName,
          loadTime: 0,
          avgDetectionTime: 0,
          avgScore: 0,
          totalDetections: 0,
          images: [],
          error: isMemoryError ? 'Out of memory' : error.message
        });
        
        // Если ошибка памяти на большой модели - пропускаем остальные большие модели
        if (isMemoryError && modelName === 'yolo11m') {
          setCurrentStatus('Memory limit reached, stopping benchmark...');
          break;
        }
      }
    }
    
    setResults(benchmarkResults);
    
    // Сохраняем сводные данные в PocketBase
    await saveBenchmarkSummary(benchmarkResults);
    
    setIsRunning(false);
    setProgress(0);
  };

  const exportResults = () => {
    const exportData = {
      device: device,
      timestamp: new Date().toISOString(),
      results: results
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `benchmark_${device}_${new Date().getTime()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div style={{ 
      marginTop: '20px', 
      padding: '20px', 
      border: '1px solid #ddd', 
      borderRadius: '8px',
      backgroundColor: '#f9f9f9'
    }}>
      <h2 style={{ marginBottom: '20px' }}>🚀 YOLO Model Benchmark</h2>
      
      {/* Информация о памяти для отладки */}
      <div style={{ 
        marginBottom: '15px', 
        padding: '8px', 
        backgroundColor: '#e9ecef', 
        borderRadius: '4px',
        fontSize: '12px'
      }}>
        <strong>Memory:</strong> {tf.memory().numTensors} tensors, {(tf.memory().numBytes / 1024 / 1024).toFixed(1)} MB
      </div>
      
      <div style={{ marginBottom: '20px', textAlign: 'center' }}>
        <button 
          onClick={runBenchmark}
          disabled={isRunning}
          style={{ 
            backgroundColor: isRunning ? '#6c757d' : '#007bff',
            color: 'white',
            padding: '12px 32px',
            border: 'none',
            borderRadius: '6px',
            fontSize: '16px',
            cursor: isRunning ? 'not-allowed' : 'pointer',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
        >
          {isRunning ? '⏳ Running Benchmark...' : '▶️ Start Benchmark'}
        </button>
        
        {results.length > 0 && (
          <button 
            onClick={exportResults}
            style={{
              backgroundColor: '#28a745',
              color: 'white',
              padding: '12px 32px',
              border: 'none',
              borderRadius: '6px',
              fontSize: '16px',
              marginLeft: '10px',
              cursor: 'pointer',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            📥 Export Results
          </button>
        )}
      </div>
      
      {isRunning && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ 
            width: '100%', 
            backgroundColor: '#e0e0e0', 
            borderRadius: '10px',
            overflow: 'hidden'
          }}>
            <div 
              style={{ 
                width: `${progress}%`, 
                backgroundColor: '#007bff', 
                height: '30px', 
                borderRadius: '10px',
                transition: 'width 0.3s',
                background: 'linear-gradient(90deg, #007bff 0%, #0056b3 100%)'
              }}
            />
          </div>
          <p style={{ textAlign: 'center', marginTop: '10px' }}>
            <strong>{progress.toFixed(1)}%</strong> - {currentStatus}
          </p>
        </div>
      )}
      
      {results.length > 0 && !isRunning && (
        <div>
          <h3>📊 Benchmark Results</h3>
          
          {/* Показываем карточки на мобильных устройствах */}
          {isMobile ? (
            <div>
              {results.map((result, index) => (
                <div key={index} style={{
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  padding: '15px',
                  marginBottom: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  border: '1px solid #dee2e6'
                }}>
                  <div style={{ 
                    fontSize: '18px', 
                    fontWeight: 'bold', 
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}>
                    <span>{result.model.toUpperCase()}</span>
                    {result.error && <span style={{ color: 'red', fontSize: '12px' }}>ERROR</span>}
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '14px' }}>
                    <div>
                      <div style={{ color: '#666', marginBottom: '4px' }}>Load Time</div>
                      <div style={{ fontWeight: 'bold' }}>{result.loadTime.toFixed(2)} ms</div>
                    </div>
                    
                    <div>
                      <div style={{ color: '#666', marginBottom: '4px' }}>Avg Detection</div>
                      <div>
                        <span style={{
                          backgroundColor: result.avgDetectionTime < 50 ? '#d4edda' : result.avgDetectionTime < 100 ? '#fff3cd' : '#f8d7da',
                          padding: '3px 6px',
                          borderRadius: '4px',
                          fontSize: '13px',
                          fontWeight: 'bold'
                        }}>
                          {result.avgDetectionTime.toFixed(2)} ms
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <div style={{ color: '#666', marginBottom: '4px' }}>Accuracy</div>
                      <div>
                        <span style={{
                          backgroundColor: result.avgScore > 0.8 ? '#d4edda' : result.avgScore > 0.6 ? '#fff3cd' : '#f8d7da',
                          padding: '3px 6px',
                          borderRadius: '4px',
                          fontSize: '13px',
                          fontWeight: 'bold'
                        }}>
                          {(result.avgScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <div style={{ color: '#666', marginBottom: '4px' }}>Detections</div>
                      <div style={{ fontWeight: 'bold' }}>{result.totalDetections}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Показываем таблицу на десктопе */
            <div style={{ overflowX: 'auto' }}>
              <table style={{ 
                width: '100%', 
                borderCollapse: 'collapse',
                backgroundColor: 'white',
                borderRadius: '8px',
                overflow: 'hidden',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8f9fa' }}>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6' }}>Model</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Load Time (ms)</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg Detection (ms)</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Avg Accuracy</th>
                    <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #dee2e6' }}>Total Detections</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => (
                    <tr key={index} style={{ borderBottom: '1px solid #dee2e6' }}>
                      <td style={{ padding: '12px', fontWeight: 'bold' }}>
                        {result.model.toUpperCase()}
                        {result.error && <span style={{ color: 'red', fontSize: '12px' }}> (Error)</span>}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>{result.loadTime.toFixed(2)}</td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>
                        <span style={{
                          backgroundColor: result.avgDetectionTime < 50 ? '#d4edda' : result.avgDetectionTime < 100 ? '#fff3cd' : '#f8d7da',
                          padding: '4px 8px',
                          borderRadius: '4px'
                        }}>
                          {result.avgDetectionTime.toFixed(2)}
                        </span>
                      </td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>
                        <span style={{
                          backgroundColor: result.avgScore > 0.8 ? '#d4edda' : result.avgScore > 0.6 ? '#fff3cd' : '#f8d7da',
                          padding: '4px 8px',
                          borderRadius: '4px'
                        }}>
                          {(result.avgScore * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>{result.totalDetections}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          <div style={{ marginTop: '20px', textAlign: 'center', color: '#6c757d' }}>
            <small>✅ Benchmark completed on {imageFiles.length} images across {models.length} models</small>
          </div>
        </div>
      )}
      
      {/* Скрытый canvas для детекции */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default ModelBenchmark;