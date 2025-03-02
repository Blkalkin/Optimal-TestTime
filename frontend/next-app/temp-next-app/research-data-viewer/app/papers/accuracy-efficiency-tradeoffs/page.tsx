'use client';

import { useEffect } from 'react';
import Link from 'next/link';

// Navigation bar component
const Navigation = () => {
  return (
    <nav className="navbar">
      <div className="container navbar-container">
        <div className="logo">
          <Link href="/">Optimal TestTime</Link>
        </div>
        <div className="nav-links">
          <Link href="/" className="nav-link">Home</Link>
          <Link href="/papers/optimal-testtime" className="nav-link">Optimal Test-Time</Link>
          <Link href="/papers/accuracy-efficiency-tradeoffs" className="nav-link active">Accuracy-Efficiency</Link>
        </div>
      </div>
    </nav>
  );
};

export default function AccuracyEfficiencyPaper() {
  // Add animation effect on scroll
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
    const sections = document.querySelectorAll('.paper-section');
    sections.forEach(section => {
      observer.observe(section);
    });

    return () => {
      sections.forEach(section => {
        observer.unobserve(section);
      });
    };
  }, []);

  return (
    <div className="paper-page">
      <Navigation />
      
      <div className="paper-header">
        <div className="container">
          <h1 className="paper-title">Accuracy-Efficiency Trade-offs in Large Language Models</h1>
          <p className="paper-subtitle">An in-depth exploration of the balance between model performance and computational efficiency</p>
          <div className="paper-meta">
            <div className="paper-authors">
              <span className="author">Vijay Kumaravel</span>
              <span className="author">David Bai</span>
              <span className="author">Balaji Kumaravel</span>
            </div>
            <div className="paper-date">May 2024</div>
          </div>
        </div>
      </div>
      
      <div className="paper-content container">
        <div className="paper-placeholder">
          <h2>Coming Soon</h2>
          <p>This paper is currently being prepared for publication. Please check back later for the full content.</p>
          <p>In the meantime, you can explore our <Link href="/papers/optimal-testtime" className="inline-link">Optimal Test-Time Computation</Link> paper.</p>
        </div>
      </div>
      
      <footer className="footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Optimal TestTime</p>
          <p>All research data available for academic and research purposes</p>
        </div>
      </footer>
    </div>
  );
} 