'use client';

import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CombinedRefusalAccuracyChart = () => {
  const [dataset, setDataset] = useState('averaged');
  
  // First dataset (test set 1)
  const firstDataset = [
    {
      model: "r1",
      total: 100,
      correct: 29,
      incorrect: 69,
      notAttempted: 2,
      refusalRate: 2.00,
      accuracyWhenAttempted: 29.59
    },
    {
      model: "r1-U",
      total: 100,
      correct: 25,
      incorrect: 63,
      notAttempted: 12,
      refusalRate: 12.00,
      accuracyWhenAttempted: 28.41
    },
    {
      model: "r1-500",
      total: 100,
      correct: 14,
      incorrect: 21,
      notAttempted: 65,
      refusalRate: 65.00,
      accuracyWhenAttempted: 40.00
    },
    {
      model: "r1-250",
      total: 100,
      correct: 5,
      incorrect: 9,
      notAttempted: 86,
      refusalRate: 86.00,
      accuracyWhenAttempted: 35.71
    },
    {
      model: "GPT-4o",
      total: 100,
      correct: 49,
      incorrect: 47,
      notAttempted: 4,
      refusalRate: 4.00,
      accuracyWhenAttempted: 51.04
    },
    {
      model: "Claude 3.5 Sonnet",
      total: 100,
      correct: 35,
      incorrect: 32,
      notAttempted: 33,
      refusalRate: 33.00,
      accuracyWhenAttempted: 52.24
    }
  ];

  // Second dataset (test set 2)
  const secondDataset = [
    {
      model: "r1",
      total: 100,
      correct: 28,
      incorrect: 61,
      notAttempted: 11,
      refusalRate: 11.00,
      accuracyWhenAttempted: 31.46
    },
    {
      model: "r1-U",
      total: 100,
      correct: 28,
      incorrect: 63,
      notAttempted: 9,
      refusalRate: 9.00,
      accuracyWhenAttempted: 30.77
    },
    {
      model: "r1-500",
      total: 100,
      correct: 20,
      incorrect: 35,
      notAttempted: 45,
      refusalRate: 45.00,
      accuracyWhenAttempted: 36.36
    },
    {
      model: "r1-250",
      total: 100,
      correct: 20,
      incorrect: 35,
      notAttempted: 45,
      refusalRate: 45.00,
      accuracyWhenAttempted: 36.36
    },
    {
      model: "GPT-4o",
      total: 100,
      correct: 44,
      incorrect: 55,
      notAttempted: 1,
      refusalRate: 1.00,
      accuracyWhenAttempted: 44.44
    },
    {
      model: "Claude 3.5 Sonnet",
      total: 100,
      correct: 26,
      incorrect: 36,
      notAttempted: 38,
      refusalRate: 38.00,
      accuracyWhenAttempted: 41.94
    }
  ];

  // Averaged dataset
  const averagedDataset = [
    {
      model: "r1",
      total: 100,
      correct: 28.5,
      incorrect: 65.0,
      notAttempted: 6.5,
      refusalRate: 6.50,
      accuracyWhenAttempted: 30.52
    },
    {
      model: "r1-U",
      total: 100,
      correct: 26.5,
      incorrect: 63.0,
      notAttempted: 10.5,
      refusalRate: 10.50,
      accuracyWhenAttempted: 29.59
    },
    {
      model: "r1-500",
      total: 100,
      correct: 17.0,
      incorrect: 28.0,
      notAttempted: 55.0,
      refusalRate: 55.00,
      accuracyWhenAttempted: 38.18
    },
    {
      model: "r1-250",
      total: 100,
      correct: 12.5,
      incorrect: 22.0,
      notAttempted: 65.5,
      refusalRate: 65.50,
      accuracyWhenAttempted: 36.03
    },
    {
      model: "GPT-4o",
      total: 100,
      correct: 46.5,
      incorrect: 51.0,
      notAttempted: 2.5,
      refusalRate: 2.50,
      accuracyWhenAttempted: 47.74
    },
    {
      model: "Claude 3.5 Sonnet",
      total: 100,
      correct: 30.5,
      incorrect: 34.0,
      notAttempted: 35.5,
      refusalRate: 35.50,
      accuracyWhenAttempted: 47.09
    }
  ];

  // Get current dataset based on selection
  const getCurrentData = () => {
    switch(dataset) {
      case 'first':
        return firstDataset;
      case 'second':
        return secondDataset;
      case 'averaged':
      default:
        return averagedDataset;
    }
  };

  const getDatasetTitle = () => {
    switch(dataset) {
      case 'first':
        return "Q100-200";
      case 'second':
        return "Q1-100";
      case 'averaged':
      default:
        return "Averaged Results";
    }
  };

  const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 rounded shadow-md">
          <p className="font-bold">{label}</p>
          <p className="text-sm">Refusal Rate: {payload[0].value.toFixed(2)}%</p>
          <p className="text-sm">Accuracy When Attempted: {payload[1].value.toFixed(2)}%</p>
          <p className="text-sm">Not Attempted: {payload[0].payload.notAttempted}</p>
          <p className="text-sm">Correct: {payload[0].payload.correct}</p>
          <p className="text-sm">Incorrect: {payload[0].payload.incorrect}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex flex-col items-center w-full p-4">
      <h2 className="text-xl font-bold mb-4">SimpleQA Hallucination Benchmark</h2>
      <h3 className="text-lg mb-3">{getDatasetTitle()}</h3>
      
      <div className="flex mb-4 space-x-2">
        <button 
          onClick={() => setDataset('second')}
          className={`px-4 py-2 rounded ${dataset === 'second' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Q1-100
        </button>
        <button 
          onClick={() => setDataset('first')}
          className={`px-4 py-2 rounded ${dataset === 'first' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Q100-200
        </button>
        <button 
          onClick={() => setDataset('averaged')}
          className={`px-4 py-2 rounded ${dataset === 'averaged' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Averaged Results
        </button>
      </div>
      
      <div className="w-full h-96">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            width={500}
            height={300}
            data={getCurrentData()}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 30,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="model" 
              height={50} 
              minTickGap={10}
            />
            <YAxis domain={[0, 100]}>
              <Label
                value="Percentage (%)"
                position="insideLeft"
                angle={-90}
                style={{ textAnchor: 'middle' }}
              />
            </YAxis>
            <Tooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: 10 }} />
            <Bar dataKey="refusalRate" name="Refusal Rate" fill="#2E86C1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="accuracyWhenAttempted" name="Accuracy When Attempted" fill="#28B463" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 p-4 bg-gray-100 rounded-lg max-w-2xl">
        <p className="italic text-gray-700">Note: All metrics are percentages from 100 test examples per dataset</p>
      </div>
    </div>
  );
};

export default CombinedRefusalAccuracyChart;
