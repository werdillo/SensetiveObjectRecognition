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
  const [memoryInfo, setMemoryInfo] = useState({ tensors: 0, bytes: 0 });
  
  const canvasRef = useRef(null);

  // Определяем мобильное устройство и возможности
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    const memoryInterval = setInterval(() => {
      const memory = tf.memory();
      setMemoryInfo({ tensors: memory.numTensors, bytes: memory.numBytes });
    }, 1000);
    
    return () => {
      window.removeEventListener('resize', checkMobile);
      clearInterval(memoryInterval);
    };
  }, []);

  // Проверка возможностей устройства
  const getDeviceCapabilities = () => {
    const memory = navigator.deviceMemory || 4;
    const isLowEnd = memory < 4 || /iPhone|iPad|Android/i.test(navigator.userAgent);
    const isIOS = /iPhone|iPad/i.test(navigator.userAgent);
    
    return {
      isLowEnd,
      isIOS,
      maxMemory: isLowEnd ? 80 * 1024 * 1024 : 200 * 1024 * 1024, // 80MB или 200MB
      gcDelay: isIOS ? 2000 : 1000
    };
  };

  const models = ['yolo11n', 'yolo11s', 'yolo11m'];
  const imageFiles = [
    'card1.png', 'card2.jpeg', 'card3.png', 'card4.jpg', 'card5.jpg',
    'card6.jpg', 'card7.jpg', 'card8.jpg', 'card9.jpg', 'card10.jpg',
    'id1.jpg', 'id2.jpg', 'id3.jpg', 'id4.jpg', 'id5.jpg',
    'id6.jpg', 'id7.jpg', 'id8.jpg', 'id9.jpg', 'id10.jpg',
    'face1.jpg', 'face2.jpg', 'face3.jpg', 'face4.jpg', 'face5.jpg',
    'face6.jpg', 'face7.jpg', 'face8.jpg', 'face9.jpg', 'face10.jpg',
    'signature1.jpg', 'signature2.jpg', 'signature3.jpg', 'signature4.jpg', 'signature5.jpg',
    'signature6.jpg', 'signature7.jpg', 'signature8.jpg', 'signature9.jpg', 'signature10.jpg',
  ];

  // Улучшенная очистка памяти
  const forceGarbageCollection = async () => {
    const capabilities = getDeviceCapabilities();
    
    // Принудительная очистка всех тензоров
    tf.dispose();
    
    // Множественные циклы GC
    const cycles = capabilities.isIOS ? 6 : 4;
    for (let i = 0; i < cycles; i++) {
      await tf.nextFrame();
    }
    
    // Дополнительная пауза для мобильных
    if (capabilities.isLowEnd) {
      await new Promise(resolve => setTimeout(resolve, capabilities.gcDelay));
    }
    
    console.log(`Memory after cleanup: ${tf.memory().numTensors} tensors, ${(tf.memory().numBytes / 1024 / 1024).toFixed(1)} MB`);
  };

  // Убираем предварительные проверки - пусть пробует загружать все модели

  const loadModel = async (modelName) => {
    try {
      setCurrentStatus(`Preparing memory for ${modelName}...`);
      
      // Очистка памяти перед загрузкой (но без проверок)
      await forceGarbageCollection();
      
      setCurrentStatus(`Loading ${modelName}...`);
      const startTime = performance.now();
      
      // Пытаемся загрузить ЛЮБУЮ модель без предварительных проверок
      const model = await Promise.race([
        tf.loadGraphModel(`/${modelName}_web_model/model.json`, {
          fetchOptions: { cache: 'no-cache' }
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Loading timeout')), 30000)
        )
      ]);
      
      setCurrentStatus(`Warming up ${modelName}...`);
      
      // Безопасный warming up
      const dummyInput = tf.ones(model.inputs[0].shape);
      const warmupResults = model.execute(dummyInput);
      
      // Очистка результатов warming up
      if (Array.isArray(warmupResults)) {
        warmupResults.forEach(tensor => tensor?.dispose?.());
      } else {
        warmupResults?.dispose?.();
      }
      dummyInput.dispose();
      
      const loadTime = performance.now() - startTime;
      
      return {
        net: model,
        inputShape: model.inputs[0].shape,
        loadTime: loadTime
      };
      
    } catch (error) {
      console.error(`Failed to load model ${modelName}:`, error);
      await forceGarbageCollection();
      throw error;
    }
  };

  const detectImage = async (image, model, canvasRef) => {
    const startTime = performance.now();
    
    try {
      const result = await Promise.race([
        detect(image, model, canvasRef.current),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Detection timeout')), 15000)
        )
      ]);
      
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
        time: performance.now() - startTime,
        score: 0,
        class: -1,
        detections: 0,
        error: error.message
      };
    }
  };

  const loadImage = (src) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const timeout = setTimeout(() => reject(new Error('Image loading timeout')), 10000);
      
      img.onload = () => {
        clearTimeout(timeout);
        resolve(img);
      };
      img.onerror = () => {
        clearTimeout(timeout);
        reject(new Error(`Failed to load ${src}`));
      };
      img.src = src;
    });
  };

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
      setCurrentStatus('Benchmark completed (save failed)');
    }
  };

  const runBenchmark = async () => {
    setIsRunning(true);
    setResults([]);
    setProgress(0);
    
    const benchmarkResults = [];
    const capabilities = getDeviceCapabilities();
    
    // Определяем какие модели тестировать
    let modelsToTest = [...models]; // копируем все модели
    
    // Если памяти мало - не тестируем yolo11m
    const memory = navigator.deviceMemory || 4;
    if (memory <= 3) {
      modelsToTest = modelsToTest.filter(model => model !== 'yolo11m');
      console.log(`Skipping yolo11m due to low memory: ${memory}GB`);
      
      // Добавляем запись о том, что модель пропущена
      benchmarkResults.push({
        model: 'yolo11m',
        loadTime: 0,
        avgDetectionTime: 0,
        avgScore: 0,
        totalDetections: 0,
        images: [],
        errors: 0,
        error: `Skipped due to insufficient RAM: ${memory}GB (minimum 4GB required)`
      });
    }
    
    // Ограничиваем изображения для слабых устройств
    const testImages = capabilities.isLowEnd ? imageFiles.slice(0, 20) : imageFiles;
    const totalTests = modelsToTest.length * testImages.length;
    let currentTest = 0;
    
    for (const modelName of modelsToTest) {
      try {
        setCurrentStatus(`Starting ${modelName}...`);
        
        const model = await loadModel(modelName);
        const modelResults = {
          model: modelName,
          loadTime: model.loadTime,
          images: [],
          totalDetections: 0,
          errors: 0
        };
        
        for (const imageFile of testImages) {
          setCurrentStatus(`Testing ${modelName} on ${imageFile}`);
          
          try {
            const image = await loadImage(`/images/${imageFile}`);
            const result = await detectImage(image, model, canvasRef);
            
            modelResults.images.push({
              imageName: imageFile,
              detectionTime: result.time,
              score: result.score,
              class: result.class,
              className: labels?.[result.class] || 'unknown',
              detections: result.detections,
              error: result.error
            });
            
            modelResults.totalDetections += result.detections;
            if (result.error) modelResults.errors++;
            
            // Периодическая очистка памяти
            if (currentTest % 10 === 0) {
              await forceGarbageCollection();
            }
            
          } catch (imgError) {
            console.error(`Error processing ${imageFile}:`, imgError);
            modelResults.errors++;
            modelResults.images.push({
              imageName: imageFile,
              detectionTime: 0,
              score: 0,
              class: -1,
              className: 'error',
              detections: 0,
              error: imgError.message
            });
          }
          
          currentTest++;
          setProgress((currentTest / totalTests) * 100);
          
          // Небольшая пауза для мобильных устройств
          if (capabilities.isLowEnd && currentTest % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 200));
          }
        }
        
        // Вычисление средних значений
        const successfulImages = modelResults.images.filter(img => !img.error);
        if (successfulImages.length > 0) {
          modelResults.avgDetectionTime = successfulImages.reduce((sum, img) => sum + img.detectionTime, 0) / successfulImages.length;
          modelResults.avgScore = successfulImages.reduce((sum, img) => sum + img.score, 0) / successfulImages.length;
        } else {
          modelResults.avgDetectionTime = 0;
          modelResults.avgScore = 0;
        }
        
        benchmarkResults.push(modelResults);
        
        // Очистка модели
        setCurrentStatus(`Cleaning up ${modelName}...`);
        if (model.net?.dispose) {
          model.net.dispose();
        }
        
        await forceGarbageCollection();
        
      } catch (error) {
        console.error(`Error with model ${modelName}:`, error);
        
        benchmarkResults.push({
          model: modelName,
          loadTime: 0,
          avgDetectionTime: 0,
          avgScore: 0,
          totalDetections: 0,
          images: [],
          errors: 1,
          error: error.message
        });
        
        // Если ошибка памяти на большой модели - прекращаем
        if (error.message.includes('memory') && modelName === 'yolo11m') {
          setCurrentStatus('Memory limit reached, stopping...');
          break;
        }
      }
    }
    
    setResults(benchmarkResults);
    await saveBenchmarkSummary(benchmarkResults);
    
    setIsRunning(false);
    setProgress(0);
    
    // Финальная очистка
    await forceGarbageCollection();
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
      
      {/* Информация о памяти */}
      <div style={{ 
        marginBottom: '15px', 
        padding: '8px', 
        backgroundColor: memoryInfo.bytes > 150 * 1024 * 1024 ? '#ffebee' : '#e8f5e8', 
        borderRadius: '4px',
        fontSize: '12px'
      }}>
        <strong>Memory:</strong> {memoryInfo.tensors} tensors, {(memoryInfo.bytes / 1024 / 1024).toFixed(1)} MB
        {memoryInfo.bytes > 150 * 1024 * 1024 && (
          <span style={{ color: 'red', marginLeft: '10px' }}>⚠️ High memory usage!</span>
        )}
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
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            marginRight: '10px'
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
          
          {isMobile ? (
            <div>
              {results.map((result, index) => (
                <div key={index} style={{
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  padding: '15px',
                  marginBottom: '15px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  border: result.error ? '2px solid #f8d7da' : '1px solid #dee2e6'
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
                  
                  {result.error ? (
                    <div style={{ color: 'red', fontSize: '14px', fontStyle: 'italic' }}>
                      {result.error}
                    </div>
                  ) : (
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
                  )}
                </div>
              ))}
            </div>
          ) : (
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
            <small>✅ Benchmark completed on {results.reduce((sum, r) => sum + r.images.length, 0)} images across {results.length} models</small>
          </div>
        </div>
      )}
      
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default ModelBenchmark;