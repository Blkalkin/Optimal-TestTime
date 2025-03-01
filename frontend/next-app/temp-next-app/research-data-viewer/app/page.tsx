'use client';

import { useEffect, useState, useRef } from 'react';
import { getResearchData, ResearchData, Dataset, Category } from '../services/dataService';

// Section component with minimalist design
const Section = ({ 
  id, 
  eyebrow, 
  title, 
  description, 
  children, 
  background = '' 
}: { 
  id: string, 
  eyebrow?: string,
  title: string,
  description?: string,
  children: React.ReactNode,
  background?: string 
}) => {
  return (
    <section id={id} className={`section ${background}`}>
      <div className="container">
        {eyebrow && <div className="section-eyebrow">{eyebrow}</div>}
        <h2 className="section-title">{title}</h2>
        {description && <p className="section-description">{description}</p>}
        {children}
      </div>
    </section>
  );
};

// Dataset card with minimal design
const DatasetCard = ({ dataset, onClick }: { 
  dataset: Dataset, 
  onClick: () => void 
}) => {
  return (
    <div className="dataset-card" onClick={onClick}>
      <h3 className="dataset-title">{dataset.title}</h3>
      <p className="dataset-description">{dataset.description}</p>
      <div className="dataset-meta">Sample size: {dataset.sampleSize}</div>
    </div>
  );
};

// Modal for dataset details
const DatasetDetail = ({ 
  dataset, 
  onClose 
}: { 
  dataset: any, 
  onClose: () => void 
}) => {
  if (!dataset) return null;
  
  return (
    <div className="modal-backdrop">
      <div className="modal-content">
        <div className="modal-header">
          <h2 className="modal-title">{dataset.title}</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <p className="modal-description">{dataset.description}</p>
          <div className="modal-meta">Sample size: {dataset.sampleSize}</div>
          
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
                {dataset.data.map((item: any) => (
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
};

// Navigation bar
const Navigation = () => {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="navbar">
      <div className="container navbar-container">
        <div className="logo">Research Data</div>
        <div className="nav-links">
          <button className="nav-link" onClick={() => scrollToSection('hero')}>Home</button>
          <button className="nav-link" onClick={() => scrollToSection('about')}>About</button>
          <button className="nav-link" onClick={() => scrollToSection('datasets')}>Data</button>
          <button className="nav-link" onClick={() => scrollToSection('visualizations')}>Insights</button>
        </div>
      </div>
    </nav>
  );
};

// Minimalist loading spinner
const LoadingSpinner = () => (
  <div className="loading-container">
    <div className="loading">
      <div className="loading-spinner"></div>
      <div>Loading data...</div>
    </div>
  </div>
);

// Error message component
const ErrorMessage = ({ message }: { message: string }) => (
  <div className="error-message">{message}</div>
);

// Bar chart visualization
const BarChart = () => {
  return (
    <div className="visualization-card">
      <div className="viz-header">
        <h3 className="viz-title">Hallucination Rates by Model</h3>
        <div className="viz-subtitle">Percentage of hallucinated responses</div>
      </div>
      <div className="viz-body">
        <div className="bar-chart">
          <div className="bar" style={{ height: '60%' }}>
            <div className="bar-value">15%</div>
            <div className="bar-label">Model A</div>
          </div>
          <div className="bar" style={{ height: '75%' }}>
            <div className="bar-value">22%</div>
            <div className="bar-label">Model B</div>
          </div>
          <div className="bar" style={{ height: '40%' }}>
            <div className="bar-value">9%</div>
            <div className="bar-label">Model C</div>
          </div>
          <div className="bar" style={{ height: '90%' }}>
            <div className="bar-value">32%</div>
            <div className="bar-label">Model D</div>
          </div>
        </div>
      </div>
      <div className="viz-footer">
        Based on analysis of 500 question-answer pairs
      </div>
    </div>
  );
};

// Donut chart visualization
const DonutChart = () => {
  return (
    <div className="visualization-card">
      <div className="viz-header">
        <h3 className="viz-title">Response Accuracy</h3>
        <div className="viz-subtitle">Overall accuracy across datasets</div>
      </div>
      <div className="viz-body">
        <div className="donut-chart">
          <div className="donut-segment" style={{ transform: 'rotate(0deg)', background: 'var(--accent-primary)' }}></div>
          <div className="donut-segment" style={{ transform: 'rotate(270deg)', background: 'var(--background-tertiary)' }}></div>
          <div className="donut-center">
            <div className="donut-value">76%</div>
            <div className="donut-label">Accuracy</div>
          </div>
        </div>
      </div>
      <div className="viz-footer">
        Based on aggregated results from all test datasets
      </div>
    </div>
  );
};

export default function Home() {
  const [data, setData] = useState<ResearchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);

  // Refs for intersection observer
  const sectionsRef = useRef<(HTMLElement | null)[]>([]);

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

  // Set up intersection observer for scroll animations
  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1,
    };

    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    
    // Get all section elements
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
      observer.observe(section);
    });

    return () => {
      sections.forEach(section => {
        observer.unobserve(section);
      });
    };
  }, [data]);

  // Function to open dataset detail modal
  const openDatasetDetail = async (categoryId: string, datasetId: string) => {
    try {
      // This would fetch the actual dataset in a real app
      const mockDataset = {
        title: `${datasetId} Dataset`.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        description: `This is the detailed view of the ${datasetId} dataset in the ${categoryId} category, containing sample responses and their evaluation scores.`,
        sampleSize: Math.floor(Math.random() * 500) + 100,
        data: Array.from({ length: 10 }, (_, i) => ({
          id: i,
          question: `Sample question ${i + 1}?`,
          answer: `Sample answer ${i + 1} for the ${datasetId} dataset.`,
          score: (Math.random() * 100).toFixed(2)
        }))
      };
      
      setSelectedDataset(mockDataset);
      document.body.style.overflow = 'hidden'; // Prevent scrolling when modal is open
    } catch (err) {
      console.error('Error loading dataset details:', err);
    }
  };

  // Function to close the dataset detail modal
  const closeDatasetDetail = () => {
    setSelectedDataset(null);
    document.body.style.overflow = 'auto'; // Re-enable scrolling
  };

  return (
    <div className="app">
      <Navigation />
      
      {/* Hero Section */}
      <section id="hero" className="hero">
        <div className="hero-content">
          <div className="hero-eyebrow">Research Data</div>
          <h1 className="hero-title">Analyzing LLM Hallucinations and Performance</h1>
          <p className="hero-subtitle">
            Explore our research data on large language model performance, hallucination rates,
            and response quality across various tasks.
          </p>
          <button 
            className="cta-button"
            onClick={() => document.getElementById('datasets')?.scrollIntoView({ behavior: 'smooth' })}
          >
            Explore Datasets
          </button>
        </div>
      </section>
      
      {/* About Section */}
      <Section 
        id="about" 
        eyebrow="Overview"
        title="Understanding Our Research" 
        description="Our work focuses on quantifying and analyzing language model behavior, with particular emphasis on hallucination detection and performance evaluation."
      >
        <div className="about-content">
          <p className="about-paragraph">
            Language models have shown remarkable capabilities across a wide range of tasks, but they also exhibit
            limitations that must be understood. Our research examines these behaviors through controlled experiments
            and comparative analysis.
          </p>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-8)', marginTop: 'var(--space-10)' }}>
            <div className="highlight-point">
              <span className="highlight-number">500+</span>
              <span className="highlight-text">Carefully crafted test questions</span>
            </div>
            <div className="highlight-point">
              <span className="highlight-number">4</span>
              <span className="highlight-text">Different LLM architectures tested</span>
            </div>
            <div className="highlight-point">
              <span className="highlight-number">76%</span>
              <span className="highlight-text">Average accuracy across models</span>
            </div>
          </div>
        </div>
      </Section>
      
      {/* Datasets Section */}
      <Section 
        id="datasets" 
        eyebrow="Data"
        title="Research Datasets" 
        description="Explore our collection of datasets used to evaluate language model performance and hallucination tendencies."
        background="bg-secondary"
      >
        {loading ? (
          <LoadingSpinner />
        ) : error ? (
          <ErrorMessage message={error} />
        ) : data ? (
          <div className="categories-container">
            {data.categories.map((category) => (
              <div key={category.id}>
                <div className="category-header">
                  <h3 className="category-title">{category.title}</h3>
                  <p className="category-description">
                    {category.id === 'hallucinations' 
                      ? 'Datasets focused on identifying and measuring hallucinations in model outputs.' 
                      : 'Evaluation of model performance across different tasks and metrics.'}
                  </p>
                </div>
                
                <div className="dataset-grid">
                  {category.datasets.map((dataset) => (
                    <DatasetCard 
                      key={dataset.id}
                      dataset={dataset}
                      onClick={() => openDatasetDetail(category.id, dataset.id)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <ErrorMessage message="No research data available." />
        )}
      </Section>
      
      {/* Visualizations Section */}
      <Section 
        id="visualizations" 
        eyebrow="Insights"
        title="Key Findings" 
        description="Visualized results from our research highlighting important patterns and discoveries."
      >
        <div className="visualization-container">
          <BarChart />
          <DonutChart />
        </div>
      </Section>
      
      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Research Data Viewer</p>
          <p>All research data available for academic and research purposes</p>
        </div>
      </footer>
      
      {/* Dataset Detail Modal */}
      {selectedDataset && (
        <DatasetDetail 
          dataset={selectedDataset} 
          onClose={closeDatasetDetail} 
        />
      )}
    </div>
  );
}