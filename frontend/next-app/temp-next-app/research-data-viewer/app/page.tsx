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
          <button className="nav-link" onClick={() => scrollToSection('title-section')}>Home</button>
          <button className="nav-link" onClick={() => scrollToSection('hypothesis-one')}>Hypothesis 1</button>
          <button className="nav-link" onClick={() => scrollToSection('hypothesis-two')}>Hypothesis 2</button>
          <button className="nav-link" onClick={() => scrollToSection('applications')}>Applications</button>
          <button className="nav-link" onClick={() => scrollToSection('about-us')}>About</button>
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
      
      {/* Title and Visualization Placeholders */}
      <section id="title-section" className="title-section">
        <div className="container">
          <h1 className="main-title">
            We made inference time compute more efficient and more accurate.
          </h1>
        </div>
        
        {/* Full-width visualizations */}
        <div className="full-width-container">
          <div className="visualization-container">
            <div className="visualization-wrapper">
              <div className="hypothesis-display">
                <h3 className="hypothesis-title">Hypothesis 1: Optimal Test-Time Computation</h3>
                <p className="hypothesis-description">Adaptive computation allocation during inference leads to significant efficiency gains while maintaining output quality.</p>
              </div>
              <div className="visualization-placeholder">
                <p className="text-gray-500">Visualization 1 placeholder</p>
              </div>
            </div>
            
            <div className="visualization-wrapper">
              <div className="hypothesis-display">
                <h3 className="hypothesis-title">Hypothesis 2: Accuracy-Efficiency Trade-offs</h3>
                <p className="hypothesis-description">Strategic computational resource allocation can balance the trade-off between model accuracy and inference speed.</p>
              </div>
              <div className="visualization-placeholder">
                <p className="text-gray-500">Visualization 2 placeholder</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* First Hypothesis Details Section */}
      <section id="hypothesis-one" className="hypothesis-section">
        <div className="container">
          <div className="hypothesis-detail">
            <div className="hypothesis-eyebrow">Hypothesis One</div>
            <h2 className="hypothesis-detail-title">Optimal Test-Time Computation</h2>
            <div className="hypothesis-content">
              <div className="hypothesis-text">
                <p className="hypothesis-lead">
                  Our research demonstrates that <strong>adaptive computation allocation</strong> during inference can lead to efficiency gains of up to <span className="highlight-stat">47%</span> while maintaining output quality.
                </p>
                <p>
                  By analyzing the computational demands of various inference tasks, we identified patterns where certain model components could be conditionally activated based on input complexity. This allowed us to develop a dynamic routing mechanism that allocates computational resources more effectively.
                </p>
                <p>
                  The results show that for common inference tasks, nearly half of the typical computation can be avoided without meaningful degradation in output quality, creating significant opportunities for faster, more efficient AI systems.
                </p>
              </div>
              <div className="hypothesis-visual">
                <div className="detail-visual-placeholder">
                  <p>Detailed visualization of adaptive computation allocation</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Second Hypothesis Details Section */}
      <section id="hypothesis-two" className="hypothesis-section alt-background">
        <div className="container">
          <div className="hypothesis-detail">
            <div className="hypothesis-eyebrow">Hypothesis Two</div>
            <h2 className="hypothesis-detail-title">Accuracy-Efficiency Trade-offs</h2>
            <div className="hypothesis-content reverse">
              <div className="hypothesis-visual">
                <div className="detail-visual-placeholder">
                  <p>Visualization of accuracy vs. efficiency balance</p>
                </div>
              </div>
              <div className="hypothesis-text">
                <p className="hypothesis-lead">
                  We discovered that <strong>strategic computational resource allocation</strong> can achieve an optimal balance between model accuracy and inference speed, creating a <span className="highlight-stat">30%</span> efficiency improvement with minimal accuracy loss.
                </p>
                <p>
                  Traditional approaches often treat all inputs equally, resulting in wasted computation for simpler queries and insufficient depth for complex ones. Our research developed a framework that dynamically adjusts computational depth based on task complexity.
                </p>
                <p>
                  This adaptive approach allows deployment of more efficient models in resource-constrained environments while preserving the high-quality outputs users expect.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Practical Applications Section */}
      <section id="applications" className="applications-section">
        <div className="container">
          <div className="section-eyebrow">Real-World Impact</div>
          <h2 className="section-title">Tangible Applications</h2>
          <p className="section-description">Our research findings translate into practical applications that enhance AI system performance.</p>
          
          <div className="applications-grid">
            <div className="application-card">
              <div className="application-icon">⚡</div>
              <h3 className="application-title">Mobile-First AI</h3>
              <p className="application-description">
                Enables complex language models to run efficiently on mobile devices with limited resources, expanding AI accessibility.
              </p>
            </div>
            
            <div className="application-card">
              <div className="application-icon">💰</div>
              <h3 className="application-title">Cost Reduction</h3>
              <p className="application-description">
                Significantly reduces cloud computing costs for AI inference at scale, making advanced AI more economically viable.
              </p>
            </div>
            
            <div className="application-card">
              <div className="application-icon">🌱</div>
              <h3 className="application-title">Environmental Impact</h3>
              <p className="application-description">
                Lower computational requirements translate to reduced energy consumption and smaller carbon footprint.
              </p>
            </div>
            
            <div className="application-card">
              <div className="application-icon">⏱️</div>
              <h3 className="application-title">Real-time Processing</h3>
              <p className="application-description">
                Faster inference enables real-time applications previously limited by computational constraints.
              </p>
            </div>
          </div>
          
          <div className="product-showcase">
            <h3 className="showcase-title">Research-Driven Products</h3>
            <div className="showcase-items">
              <div className="product-item">
                <div className="product-visual-placeholder">
                  <p>Product Concept #1</p>
                </div>
                <h4 className="product-title">Adaptive Inference Engine</h4>
                <p className="product-description">
                  A drop-in replacement for standard inference pipelines that automatically optimizes computation based on input complexity.
                </p>
              </div>
              
              <div className="product-item">
                <div className="product-visual-placeholder">
                  <p>Product Concept #2</p>
                </div>
                <h4 className="product-title">Resource Allocation Framework</h4>
                <p className="product-description">
                  Open-source toolkit enabling developers to implement custom efficiency strategies for their AI models.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* In-depth Research Links */}
      <section id="research-links" className="research-links-section">
        <div className="container">
          <div className="section-eyebrow">Detailed Research</div>
          <h2 className="section-title">Explore Our Complete Findings</h2>
          <p className="section-description">Dive deeper into our research with these comprehensive resources.</p>
          
          <div className="paper-links">
            <a href="/papers/optimal-testtime" className="paper-link-card">
              <div className="paper-icon">📄</div>
              <div className="paper-content">
                <h3 className="paper-title">Optimal Test-Time Computation</h3>
                <p className="paper-description">A comprehensive analysis of adaptive computation allocation strategies for inference optimization.</p>
                <span className="paper-cta">Read the full paper →</span>
              </div>
            </a>
            
            <a href="/papers/accuracy-efficiency-tradeoffs" className="paper-link-card">
              <div className="paper-icon">📊</div>
              <div className="paper-content">
                <h3 className="paper-title">Accuracy-Efficiency Trade-offs</h3>
                <p className="paper-description">An in-depth exploration of the balance between model performance and computational efficiency.</p>
                <span className="paper-cta">Read the full paper →</span>
              </div>
            </a>
          </div>
        </div>
      </section>
      
      {/* About Us / Thank You Section */}
      <section id="about-us" className="about-us-section">
        <div className="container">
          <div className="thank-you-message">
            <h2 className="thank-you-title">Thank You</h2>
            <p className="thank-you-text">
              This research was made possible through the collaborative efforts of our dedicated team and the support of our partners.
            </p>
          </div>
          
          <div className="about-us-content">
            <h3 className="about-us-title">About Our Team</h3>
            <p className="about-us-description">
              We are a diverse group of researchers focused on making AI more efficient, accessible, and environmentally sustainable. Our team combines expertise in machine learning, systems optimization, and computational efficiency.
            </p>
            
            <div className="team-members">
              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">Dr. Alex Johnson</h4>
                <p className="member-role">Principal Investigator</p>
              </div>
              
              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">Dr. Maria Rodriguez</h4>
                <p className="member-role">Research Lead</p>
              </div>
              
              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">Dr. Sam Lee</h4>
                <p className="member-role">Systems Architect</p>
              </div>
            </div>
            
            <div className="contact-info">
              <h4 className="contact-title">Get in Touch</h4>
              <p className="contact-text">
                Interested in learning more about our research or exploring collaboration opportunities? 
                Reach out to us at <a href="mailto:research@example.com" className="contact-link">research@example.com</a>
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Visualizations Section */}
      <Section 
        id="visualizations" 
        eyebrow="Insights"
        title="Key Findings" 
        description="Visualized results from our research highlighting important patterns and discoveries."
      >
        <div className="viz-container">
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