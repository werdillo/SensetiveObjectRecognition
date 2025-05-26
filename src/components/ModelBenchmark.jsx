import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { detect } from '../utils/detect';
import { client } from '../utils/pocketbase';

const ModelBenchmark = ({ device, client, labels }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState('');
  const [results, setResults] = useState([]);
  const canvasRef = useRef(null);

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
    // Добавьте сюда имена ваших тестовых изображений
  ];

  const loadModel = async (modelName) => {
    const startTime = performance.now();
    
    const model = await tf.loadGraphModel(
      `/${modelName}_web_model/model.json`
    );
    
    // Warming up
    const dummyInput = tf.ones(model.inputs[0].shape);
    const warmupResults = model.execute(dummyInput);
    tf.dispose([warmupResults, dummyInput]);
    
    const loadTime = performance.now() - startTime;
    
    return {
      net: model,
      inputShape: model.inputs[0].shape,
      loadTime: loadTime
    };
  };

  const detectImage = async (image, model, canvasRef) => {
    const startTime = performance.now();
    
    // Используем вашу функцию detect
    const result = await detect(image, model, canvasRef.current);
    
    const detectionTime = performance.now() - startTime;
    
    return {
      time: detectionTime,
      score: result.scores?.[0] || 0,
      class: result.classes?.[0] || -1,
      detections: result.scores?.length || 0
    };
  };

  const loadImage = (src) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(`Failed to load ${src}`);
      img.src = src;
    });
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
      setCurrentStatus(`Loading model: ${modelName}`);
      
      try {
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
            
            // Убираем сохранение индивидуальных результатов - сохраняем только итоговую сводку
            // if (client) {
            //   await client.collection('benchmarkResults').create({
            //     device: device,
            //     model: modelName,
            //     imageName: imageFile,
            //     detectionTimeMs: result.time.toFixed(2),
            //     score: result.score.toFixed(4),
            //     class: imageResult.className,
            //     detections: result.detections,
            //     timestamp: new Date().toISOString()
            //   });
            // }
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
        
        // Очистка модели
        model.net.dispose();
        
      } catch (error) {
        console.error(`Error with model ${modelName}:`, error);
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
                  <td style={{ padding: '12px', fontWeight: 'bold' }}>{result.model.toUpperCase()}</td>
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