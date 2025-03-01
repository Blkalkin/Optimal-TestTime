'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getDatasetDetails } from '../../../../services/dataService';

interface DatasetDetails {
  title: string;
  description: string;
  sampleSize: number;
  data: any[]; // This would be more specifically typed in a real application
}

export default function DatasetPage() {
  const params = useParams();
  const { categoryId, datasetId } = params;
  
  const [loading, setLoading] = useState(true);
  const [dataset, setDataset] = useState<DatasetDetails | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDatasetDetails() {
      try {
        if (typeof categoryId !== 'string' || typeof datasetId !== 'string') {
          throw new Error('Invalid category or dataset ID');
        }
        
        const details = await getDatasetDetails(categoryId, datasetId);
        setDataset(details);
      } catch (err) {
        setError('Failed to load dataset details');
        console.error('Error fetching dataset details:', err);
      } finally {
        setLoading(false);
      }
    }

    if (categoryId && datasetId) {
      fetchDatasetDetails();
    }
  }, [categoryId, datasetId]);

  if (loading) {
    return (
      <div className="loading">
        <div className="pulse">Loading dataset...</div>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="container">
        <div className="error">
          {error || 'Failed to load dataset'}
        </div>
        <Link href="/" className="mt-4">
          Return to home
        </Link>
      </div>
    );
  }

  return (
    <div className="main">
      <div className="container">
        <div className="mb-4">
          <Link href="/" className="mb-4">
            &larr; Back to Datasets
          </Link>
          <h1 className="title">{dataset.title}</h1>
          <p className="mb-2">{dataset.description}</p>
          <div className="text-sm mb-4">Sample size: {dataset.sampleSize}</div>
          
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Question</th>
                  <th>Answer</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {dataset.data.map((item) => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.question}</td>
                    <td>{item.answer}</td>
                    <td>{item.score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}