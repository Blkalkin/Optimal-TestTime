'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getResearchData, ResearchData } from '../services/dataService';

export default function Home() {
  const [data, setData] = useState<ResearchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const researchData = await getResearchData();
        setData(researchData);
      } catch (error) {
        console.error('Error fetching research data:', error);
        setError('Failed to load research data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <main className="main">
      <div className="container">
        <div className="header">
          <h1 className="title">Research Data Viewer</h1>
          <p className="text-xl mb-4">Explore and visualize research datasets</p>
        </div>
        
        {loading ? (
          <div className="loading">
            <div className="pulse">Loading data...</div>
          </div>
        ) : error ? (
          <div className="error">
            {error}
          </div>
        ) : data ? (
          <div>
            {data.categories.map((category) => (
              <div key={category.id} className="mb-4">
                <h2 className="text-2xl font-bold mb-2">{category.title}</h2>
                
                <div className="grid">
                  {category.datasets.map((dataset) => (
                    <Link 
                      href={`/datasets/${category.id}/${dataset.id}`} 
                      key={dataset.id}
                      className="card"
                    >
                      <h3 className="card-title">{dataset.title}</h3>
                      <p className="card-description">{dataset.description}</p>
                      <div className="card-meta">Sample size: {dataset.sampleSize}</div>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="error">
            No research data available.
          </div>
        )}
      </div>
    </main>
  );
}