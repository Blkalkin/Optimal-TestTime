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
        <div className="logo">Optimal TestTime</div>
        <div className="nav-links">
          <button className="nav-link" onClick={() => scrollToSection('title-section')}>Home</button>
          <button className="nav-link" onClick={() => scrollToSection('hypothesis-one')}>Efficiency</button>
          <button className="nav-link" onClick={() => scrollToSection('hypothesis-two')}>Accuracy</button>
          <button className="nav-link" onClick={() => scrollToSection('applications')}>Applications</button>
          <button className="nav-link" onClick={() => scrollToSection('research-links')}>Research</button>
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
                <h3 className="hypothesis-title">Efficiency: Parallel Reasoning Chain Pruning</h3>
                <p className="hypothesis-description">By evaluating the semantic similarity between reasoning chains at various stages and pruning similar paths while introducing diverse alternatives, we can achieve the same accuracy benefits of best-of-10^n approaches at significantly reduced computational cost.</p>
              </div>
              <div className="visualization-placeholder">
                <p className="text-gray-500">Visualization 1 placeholder</p>
              </div>
            </div>
            
            <div className="visualization-wrapper">
              <div className="hypothesis-display">
                <h3 className="hypothesis-title">Accuracy: Inference-Time Hallucination Detection via Self-Verification</h3>
                <p className="hypothesis-description">By having reasoning LLMs extend their thinking with self-verification statements after reaching an answer, we can implement a majority-vote mechanism to detect hallucinations, enabling confidence-based compute allocation for improved accuracy on various benchmarks.</p>
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
            <div className="hypothesis-eyebrow">Efficiency</div>
            <h2 className="hypothesis-detail-title">Parallel Reasoning Chain Pruning</h2>
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
            <div className="hypothesis-eyebrow">Accuracy</div>
            <h2 className="hypothesis-detail-title">Inference-Time Hallucination Detection via Self-Verification</h2>
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
          <p className="section-description">Our optimal test-time computation research potentially addresses critical challenges in AI deployment.</p>
          
          <div className="gpu-shortage-container">
            <div className="gpu-shortage-content">
              <h3 className="shortage-title">We're out of GPUs</h3>
              <p className="shortage-description">
                As AI capabilities expand, we're facing unprecedented demand for compute. Our research could offer optimizations that help alleviate this bottleneck.
              </p>
              <div className="key-benefits">
                <div className="benefit-item">
                  <span className="benefit-icon">💰</span>
                  <span className="benefit-text"><strong>40% cost reduction</strong> through strategic resource allocation</span>
                </div>
                <div className="benefit-item">
                  <span className="benefit-icon">🌱</span>
                  <span className="benefit-text"><strong>Significantly lower</strong> energy consumption and carbon footprint</span>
                </div>
                <div className="benefit-item">
                  <span className="benefit-icon">🛡️</span>
                  <span className="benefit-text"><strong>Enhanced reliability</strong> through reduced hallucinations</span>
                </div>
              </div>
            </div>
            <div className="gpu-shortage-tweet">
              <div className="tweet-placeholder">
                <p className="tweet-content">"Sam Altman tweet"</p>
                <p className="tweet-author">— Sam Altman</p>
              </div>
            </div>
          </div>
          
          <div className="company-impact">
            <h3 className="impact-title">Industry Applications</h3>
            <p className="impact-description">
              Our research enables breakthrough capabilities for leading AI innovators:
            </p>
            
            <div className="impact-items">
              <div className="impact-item">
                <h4 className="impact-company">Etched → Sohu</h4>
                <p className="impact-text">
                  Sohu, the world's first specialized chip (ASIC) for transformers, could leverage our "parallel reasoning chain pruning" to maximize throughput. By "pruning", Sohu can match the ever-increasing demand for AI compute with significantly higher efficiency than general-purpose GPUs, leading to new paradigms not currently imaginable.
                </p>
              </div>
              
              <div className="impact-item">
                <h4 className="impact-company">Cognition (Devin)</h4>
                <p className="impact-text">
                  Devin, the AI software engineer, requires high factual accuracy and efficient resource usage when working across large codebases, while not generating any errors. Our refusal research could enable more reliable code generation, making autonomous coding systems like Devin more powerful.
                </p>
              </div>
              
              <div className="impact-item">
                <h4 className="impact-company">Mercor</h4>
                <p className="impact-text">
                  For AI-driven development platforms like Mercor, our "refusal research could enhance the accuracy of technical solutions while maintaining responsiveness, while our "pruning research" could enable "n-50 etc".
                </p>
              </div>
            </div>
          </div>
          
          <div className="product-showcase">
            <h3 className="showcase-title">Research-Driven Products</h3>
            <div className="showcase-items">
              <div className="product-item">
                <div className="product-visual-placeholder">
                  <p>World Models</p>
                </div>
                <h4 className="product-title">World Model Optimization</h4>
                <p className="product-description">
                  Our research could enable more efficient world models that can simulate complex environments with selective computation, focusing resources on the most informative aspects of the simulation while maintaining fidelity. S/O @etched.
                </p>
              </div>
              
              <div className="product-item">
                <div className="product-visual-placeholder">
                  <p>Generalist Agent</p>
                </div>
                <h4 className="product-title">Generalist Autonomous Assistant</h4>
                <p className="product-description">
                  Our research could enable generalist autonomous agents that can reason across multiple contexts simultaneously while maintaining accuracy through strategic optimial test-time computation.
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
              This research was made possible through the Cognition X Mercor X Etched Hackathon.
              We want to thank @Coreweave through @Northflank for providing acces to 8 x H100s and their team for trouble shooting with us.
              Thank you Mercor for the office space and organizing the event.
              Thanks to @Anthropic for the credits to use Claude Code and Sonnet, and to @Cognition for Devin access.
              Special thanks to @Etched for extremely interesting conversations and guidance.
            </p>
          </div>
          
          <div className="about-us-content">
            <h3 className="about-us-title">About Our Team</h3>
            <p className="about-us-description">
              We like AI. We love trying to make it better.
            </p>
            
            <div className="team-members">
              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">Vijay Kumaravel</h4>
                <p className="member-role">Cofounder @JargonLearn & @Empathy, Researcher/Junior @USC</p>
              </div>
              
              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">David Bai</h4>
                <p className="member-role">Cofounder @JargonLearn & @Empathy, Researcher/Sophmore @USC</p>
              </div>

              <div className="team-member">
                <div className="member-placeholder"></div>
                <h4 className="member-name">Balaji Kumaravel</h4>
                <p className="member-role">Founding Engineer @Adapt API, ex-Quantative Trading Engineer</p>
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
      
      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Optimal TestTime</p>
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