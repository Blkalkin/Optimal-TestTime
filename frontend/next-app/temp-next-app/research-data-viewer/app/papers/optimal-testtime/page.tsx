'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';

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
          <Link href="/papers/optimal-testtime" className="nav-link active">Parallel Reasoning Pruning</Link>
          <Link href="/papers/accuracy-efficiency-tradeoffs" className="nav-link">Hallucination Detection</Link>
        </div>
      </div>
    </nav>
  );
};

// Paper section component
const PaperSection = ({ 
  id, 
  title, 
  children 
}: { 
  id: string, 
  title: string,
  children: React.ReactNode
}) => {
  return (
    <section id={id} className="paper-section">
      <h2 className="paper-section-title">{title}</h2>
      <div className="paper-section-content">
        {children}
      </div>
    </section>
  );
};

// Figure component for displaying images with captions
const Figure = ({
  src,
  alt,
  title,
  caption
}: {
  src: string,
  alt: string,
  title: string,
  caption: string
}) => {
  return (
    <div className="figure-container">
      <h3 className="figure-title">{title}</h3>
      <div className="figure-image">
        <Image 
          src={src}
          alt={alt}
          width={800}
          height={500}
          style={{
            width: '100%',
            height: 'auto',
            objectFit: 'contain'
          }}
        />
      </div>
      <p className="figure-caption">{caption}</p>
    </div>
  );
};

export default function OptimalTestTimePaper() {
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
          <h1 className="paper-title">Improving Inference Efficiency through Pruning of Parallel Reasoning Chains</h1>
          <p className="paper-subtitle">A comprehensive analysis of adaptive computation allocation strategies for inference optimization</p>
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
        <div className="paper-toc">
          <h3>Table of Contents</h3>
          <ul>
            <li><a href="#abstract">Abstract</a></li>
            <li><a href="#model">Model</a></li>
            <li><a href="#metrics">Metrics</a></li>
            <li><a href="#benchmark">Benchmark</a></li>
            <li><a href="#infrastructure">Infrastructure</a></li>
            <li><a href="#pipeline">Pipeline</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#comparison">Comparison to Baseline</a></li>
            <li><a href="#limitations">Limitations and Future Work</a></li>
          </ul>
        </div>
        
        <div className="paper-body">
          <PaperSection id="abstract" title="Abstract">
            <p>
              Inference time scaling has proven extremely effective at improving LLM performance in domains with a generator-verifier gap, 
              where generating candidate solutions is much harder than verifying correctness. Several popular methodologies for scaling 
              inference compute have been explored, with many widely used approaches involving Reinforcement Learning to elicit long 
              Chains-Of-Thought for self-correction, as well as generating multiple candidate solutions and selecting the most correct 
              one (known as best-of-n). Combining these methodologies has proven highly effective, boosting key benchmark results in 
              competitive coding (IOI for o3) and mathematics (Frontier Math, AIME).
            </p>
            <p>
              This paper explores a more inference-efficient approach to scaling best-of-n for reasoning models through parallel reasoning, 
              by pruning reasoning chains early when they don't contribute to candidate solution diversity. Our experiments on the AIME 
              competition math benchmark demonstrate that our method achieves equivalent pass@50 performance by pruning 40 reasoning chains 
              after only 300 tokens, decoding just 10 reasoning chains to completion.
            </p>
          </PaperSection>
          
          <PaperSection id="model" title="Model">
            <p>
              For our experiments, we selected DeepSeek-R1-Distill-Llama-70B for two key reasons:
            </p>
            <ol>
              <li>
                It is a distillation of the full DeepSeek-R1-671B, offering similar performance on reasoning benchmarks 
                while being much more efficient to serve, making it a de-facto cost-effective reasoning model in industry.
              </li>
              <li>
                It allowed for more extensive data collection under the constraints of our 24-hour hackathon with an 8xH100 node.
              </li>
            </ol>
          </PaperSection>
          
          <PaperSection id="metrics" title="Metrics">
            <p>
              The primary metric in this work is pass@k, which measures how many questions a model answers correctly in a benchmark given k attempts.
            </p>
          </PaperSection>
          
          <PaperSection id="benchmark" title="Benchmark">
            <p>
              Our experiments focused on the AIME competition math benchmark for several reasons:
            </p>
            <ol>
              <li>
                Competition mathematics problems elicit long reasoning chains, which was crucial for determining 
                the optimal pruning point and demonstrating maximum compute savings.
              </li>
              <li>
                Competition math is not completely saturated by our chosen model, providing clearer signals about 
                whether our method preserves performance.
              </li>
              <li>
                AIME problems have single numerical answers, simplifying extraction and verification for our evaluation pipelines.
              </li>
            </ol>
          </PaperSection>
          
          <PaperSection id="infrastructure" title="Infrastructure">
            <p>
              We conducted our experiments on an 8xH100 node, generously provided by CoreWeave and North Flank for 24 hours. 
              We used vLLM as the inference engine to run DeepSeek-R1-Distill-Llama-70B with a temperature of 0.7, top p of 0.9, 
              and a batch size of 50 per prompt. We collected as much data as possible within the time constraints to enhance 
              the confidence in our results.
            </p>
          </PaperSection>
          
          <PaperSection id="pipeline" title="Pipeline">
            <p>
              Our experimental pipeline consisted of the following steps:
            </p>
            <ol>
              <li>Generate 50 reasoning chains per AIME problem using DeepSeek-R1-Distill-Llama-70B.</li>
              <li>
                Based on an initial experimental pass, we found the median token length for AIME problems was 23k, 
                so we set our max_output tokens to 23k for efficient batch inference.
              </li>
              <li>
                We collected data for all problems in the 2023 AIME part 2 and the 2024 AIME part 1, along with a 
                validation set from 2023 AIME part 1.
              </li>
              <li>
                We used GPT-4o to extract answers from each response and determined the baseline pass@50 performance. 
                With 50 attempts, the model had at least one correct answer for 26/29 problems (89%).
              </li>
            </ol>

            <Figure 
              src="/images/papers/optimal-testtime/token_distribution.png"
              alt="Distribution of token counts in responses"
              title="Distribution of Token Counts in Responses"
              caption="Distribution of token lengths across model responses, showing a median of 22,043 tokens."
            />

            <p>
              After collecting this data, we proceeded with the following additional steps:
            </p>
            <ol start={5}>
              <li>We chunked the reasoning chains into sections of 300 tokens using the Cl100k_base tokenizer.</li>
              <li>These chunks were fed into the OpenAI Large v3 embedding model to generate embeddings for clustering.</li>
              <li>
                We conducted a hyperparameter sweep to find the optimal pruning point and number of chains to retain, 
                testing 7 different values (10, 15, 20, 25, 30, 35, 40) and evaluating at every chunk up to the 
                halfway point of the reasoning chain.
              </li>
            </ol>
          </PaperSection>
          
          <PaperSection id="results" title="Results">
            <p>
              The results of our hyperparameter sweep are presented in the table below:
            </p>
            
            <div className="table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Chunk Index</th>
                    <th>10</th>
                    <th>15</th>
                    <th>20</th>
                    <th>25</th>
                    <th>30</th>
                    <th>35</th>
                    <th>40</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>0</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                  </tr>
                  <tr>
                    <td>1</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                  </tr>
                  <tr>
                    <td>2</td>
                    <td>86%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                  </tr>
                  <tr>
                    <td>3</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>89%</td>
                    <td>89%</td>
                    <td>89%</td>
                  </tr>
                  <tr>
                    <td>4</td>
                    <td>75%</td>
                    <td>79%</td>
                    <td>82%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                  </tr>
                  <tr>
                    <td>5</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>86%</td>
                  </tr>
                  <tr>
                    <td>6</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                  </tr>
                  <tr>
                    <td>7</td>
                    <td>79%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                  </tr>
                  <tr>
                    <td>8</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>86%</td>
                    <td>82%</td>
                  </tr>
                  <tr>
                    <td>9</td>
                    <td>75%</td>
                    <td>75%</td>
                    <td>82%</td>
                    <td>82%</td>
                    <td>75%</td>
                    <td>79%</td>
                    <td>79%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <p>
              We found that using embeddings from just the first 1-3 chunks of the reasoning chains for clustering and pruning 
              resulted in performance equal to allowing all 50 reasoning chains to decode to completion. While these results 
              are subject to limitations in our benchmark scope, they are promising and warrant further exploration.
            </p>
          </PaperSection>
          
          <PaperSection id="comparison" title="Comparison to Baseline">
            <p>
              The appropriate baseline for comparison is the pass@10 performance of the model, as the compute requirements 
              are essentially equivalent to our pruned pass@50 approach.
            </p>

            <Figure 
              src="/images/papers/optimal-testtime/original_custom_pass_at_k.png"
              alt="Baseline pass@k Performance vs Pruning Strategy"
              title="Baseline pass@k Performance vs Pruning Strategy"
              caption="Performance comparison on AIME 2023 I and 2024 I, showing optimal compute efficiency at k=10."
            />

            <Figure 
              src="/images/papers/optimal-testtime/val_custom_pass_at_k.png"
              alt="Performance comparison on AIME 2023 II"
              title="Performance Comparison on AIME 2023 II"
              caption="Performance comparison on AIME 2023 II, showing optimal compute efficiency at k=10."
            />
          </PaperSection>
          
          <PaperSection id="limitations" title="Limitations and Future Work">
            <p>
              We plan to test this methodology on additional reasoning benchmarks, such as competitive coding and chess puzzles 
              with efficient verifiers. We would also like to implement this method on hardware and measure its performance 
              compared to the baseline via FLOP utilization. Currently, clustering of reasoning chains is done through k-means, 
              but other methods for preserving diverse reasoning chains could include using BLEU, ROUGE, and BLEURT scores.
            </p>
          </PaperSection>
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