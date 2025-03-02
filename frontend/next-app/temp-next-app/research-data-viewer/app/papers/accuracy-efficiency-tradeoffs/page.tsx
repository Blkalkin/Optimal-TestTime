'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import CombinedRefusalAccuracyChart from "../../../components/CombinedRefusalAccuracyChart";

// Figure component for displaying images with captions
const Figure = ({ 
  src, 
  alt, 
  title, 
  caption,
  width,
  height 
}: { 
  src: string; 
  alt: string; 
  title: string; 
  caption: string;
  width?: string | number;
  height?: string | number;
}) => {
  return (
    <figure className="paper-figure">
      <img 
        src={src} 
        alt={alt} 
        className="paper-image" 
        style={width || height ? { width: width || 'auto', height: height || 'auto' } : undefined} 
      />
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-caption">{caption}</span>
      </figcaption>
    </figure>
  );
};

// Section component for paper sections with animations
const PaperSection = ({ id, title, children }: { id: string; title: string; children: React.ReactNode }) => {
  return (
    <section id={id} className="paper-section">
      <h2 className="section-title">{title}</h2>
      <div className="section-content">
        {children}
      </div>
    </section>
  );
};

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
          <Link href="/papers/accuracy-efficiency-tradeoffs" className="nav-link">Accuracy-Efficiency</Link>
        </div>
      </div>
    </nav>
  );
};

export default function InterruptionPaper() {
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

  // Track active section for table of contents
  const [activeSection, setActiveSection] = useState('abstract');

  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll('.paper-section');
      const scrollPosition = window.scrollY + 100;

      sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');

        if (
          scrollPosition >= sectionTop &&
          scrollPosition < sectionTop + sectionHeight && 
          sectionId
        ) {
          setActiveSection(sectionId);
        }
      });
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <div className="paper-page">
      <Navigation />
      
      <div className="paper-header">
        <div className="container">
          <h1 className="paper-title">Interruption is All You Need: Improving Reasoning Model Refusal Rates through measuring Parallel Reasoning Diversity</h1>
          <p className="paper-subtitle">A novel approach to reducing hallucinations in large language models through parallel reasoning and diversity measurement</p>
          <div className="paper-meta">
            <div className="paper-authors">
              <span className="author">David Bai</span>
              <span className="author">Vijay Kumaravel</span>
              <span className="author">Balaji Kumaravel</span>
            </div>
            <div className="paper-date">June 2024</div>
          </div>
        </div>
      </div>
      
      <div className="paper-content container">
        {/* Use paper-toc instead of custom table-of-contents */}
        <div className="paper-toc">
          <h3>Table of Contents</h3>
          <ul>
            <li className={activeSection === 'abstract' ? 'active' : ''}>
              <a href="#abstract">Abstract</a>
            </li>
            <li className={activeSection === 'introduction' ? 'active' : ''}>
              <a href="#introduction">Introduction</a>
            </li>
            <li className={activeSection === 'method' ? 'active' : ''}>
              <a href="#method">Method</a>
            </li>
            <li className={activeSection === 'experiment-details' ? 'active' : ''}>
              <a href="#experiment-details">Experiment Details</a>
            </li>
            <li className={activeSection === 'results' ? 'active' : ''}>
              <a href="#results">Results</a>
            </li>
            <li className={activeSection === 'conclusion' ? 'active' : ''}>
              <a href="#conclusion">Conclusion</a>
            </li>
            <li className={activeSection === 'appendix' ? 'active' : ''}>
              <a href="#appendix">Appendix</a>
            </li>
          </ul>
        </div>
        
        <div className="paper-body">
          <PaperSection id="abstract" title="Abstract">
            <p>
              We propose a method for increasing LLM refusal rates on questions they'd typically get incorrect by utilizing inference-time compute to scale reasoning, 
              then verifying the diversity of reasoning chains that arrive towards a coherent answer. Given a query, we run inference on Deepseek R1, then interrupt it 
              at N tokens, injecting an interruption token: "No, but". We then run P parallel inferences from that Nth token, allowing us to sample different reasoning 
              traces given the same interruption token. Once R1 has resolved to an answer, we utilize an SLM to judge the coherence and diversity of the P reasoning 
              traces with a score. If this score exceeds a tuned threshold, we choose to reject the original LLM answer and choose not to attempt to answer the respective question.
            </p>
            <p>
              We find this method maintains accuracy while increasing refusal rates on incorrect answers, with further work necessary to derive the optimum Nth token 
              at which to inject. We believe this method is highly applicable to deployments and spaces where false negatives are highly consequential— such as high-trust 
              environments like the medical or legal field, or in spaces where LLM outputs are too large to be human-verified, where this method is a form of scaleable oversight.
            </p>
          </PaperSection>
          
          <PaperSection id="introduction" title="Introduction">
            <p>
              Reasoning models leverage chain of thought to reason through and perform extremely well on verifiable tasks like coding and math, 
              which require multi-hop reasoning and thinking over long contexts. This use of inference-time compute is typically emergent through 
              reinforcement learning, as demonstrated by models like Deepseek-R1 and the slew of replications that followed.
            </p>
            <p>
              However, this reinforcement learning and emergent behavior often means that these models, on tasks like retrieval or memorization 
              (hallucination benchmarks like SimpleQA), attempt to "reason through" this sort of question. While it's possible that the model 
              may reason over other context it has, the majority of these questions result in overconfident hallucination, especially in the 
              case of Deepseek-R1, which is the model we tested our method on.
            </p>
            <p>
              Prior work, like s1 from Stanford and other papers pre-reasoning paradigm, has demonstrated that we can "budget force" a model by 
              injecting tokens like "Wait" or even just periods. We aim to use this to gauge model uncertainty.
            </p>
          </PaperSection>
          
          <PaperSection id="method" title="Method">
            <p>
              Given a query Q, we run inference on a reasoning model M, until an EOS and answer is produced. Additionally, at hyperparameter N 
              tokens in the same generated sequence, we inject an interrupt token "No, but" (though we'd like to try other examples) to induce 
              uncertainty into the reasoning trace. We then continue this reasoning trace in hyperparameter P parallel instances until they all terminate.
            </p>
            <p>
              We then prompt a SLM (Small Language Model) to judge the diversity of these reasoning traces— how different they are, whether they 
              are coherent with each other. We find that a scalar for diversity from 1-10 is expressive enough for our purposes, as opposed to 
              a regression or a larger range.
            </p>
            <p>
              Then, for Q, if the diversity score exceeds our set threshold, we make the answer a refusal, e.g "I don't know", which is classified 
              by our judge benchmark as NO_ATTEMPT.
            </p>
            
            <Figure 
              src="/images/papers/interruption-paper/method-diagram.png"
              alt="Method diagram showing interruption and parallel reasoning"
              title="Figure 1: Method Overview"
              caption="The interruption method process: inference until N tokens, inject 'No, but', branch into P parallel reasoning paths, and measure diversity for refusal decision."
            />
          </PaperSection>
          
          <PaperSection id="experiment-details" title="Experiment Details">
            <h3>Benchmark: SimpleQA</h3>
            <p>
              We utilize OpenAI's SimpleQA, a benchmark for measuring language model factuality, featuring 4,326 diverse fact-seeking questions 
              with verified answers, where model responses are classified as "correct," "incorrect," or "not attempted" using a prompted classifier, 
              specifically designed to identify and reduce hallucinations in AI-generated content.
            </p>
            <p>
              Given budget and compute constraints, we form two sets, the first 100 and second 100 questions of SimpleQA to evaluate on, and report 
              results for both.
            </p>
            
            <h3>Model: Deepseek R1</h3>
            <p>
              We utilize Deepseek R1 for this because of its open-source reasoning traces and reported ability to self-correct, as well as 
              the ability to insert prefills into its thinking tokens, crucial to our method. For this paper, all results are from the full 
              671B version of Deepseek R1. We run Deepseek R1 using the Fireworks API, with a temperature of 1.0 and a top p of 1.
            </p>
            
            <h3>Answer Diversity through SLMs</h3>
            <p>
              An initial attempt we tried, for finding consensus from parallel reasoning chains (the original motivation for this work) was 
              to simply measure the answer diversity after the reasoning chains had concluded (and the answer was extracted from the response 
              after the &lt;/think&gt; token). We discovered that the reasoning model often arrived at the same conclusion, with fairly different 
              reasoning chains or rationales. Thus, this did not work on our preliminary experiments.
            </p>
            <p>
              Because we ultimately did not follow through with this method, we do not include its results in the Results Section.
            </p>
            
            <h3>Continuation Diversity through Embeddings</h3>
            <p>
              The next intuition we had was that if many different reasoning chains were being used to arrive to the same conclusion, 
              then that conclusion could likely be false. Or in the rare case, supported by many different sources and thus true. Our 
              first attempt at measuring reasoning chain diversity (solely the sequence starting from the injected interruption token) 
              was to use embeddings, specifically OpenAI's text-embedding-3-small. What we discovered was that similarity rates were 
              often very high, and the embeddings did not capture the nuance we wanted. When comparing 2 large-token chains-of-thought 
              with just 1 or 2 differences, its likely that the embedding cosine similarity (what we used to measure diversity) will be 
              very high even if those changes result in very different reasoning traces.
            </p>
            <p>
              Because we ultimately did not follow through with this method, we do not include its results in the Results Section. We 
              include details and figures about this in the appendix.
            </p>
            
            <h3>Reasoning Diversity though SLMs</h3>
            <p>
              We ultimately decided to use a SLM to judge reasoning chain diversity, using Gemini 2.0 Flash-Lite as a proxy instead of 
              locally hosting a solution, as our compute node was being fully utilized by our other experiments for another project. 
              The SLM is given all of the parallel reasoning chains, and asked to rate their diversity on a scale from 1-10.
            </p>
          </PaperSection>
          
          <PaperSection id="results" title="Results">
            <p>
              We utilize GPT-4o and Claude 3.5 Sonnet as our baselines for their strong performance and high deployment rate in real-world 
              scenarios. For our method, we report three results, each one representing the method applied on an injection token at 250 tokens, 
              500 tokens, and unbounded (10,000 but never reached) tokens. We find that our average token expenditure for SimpleQA questions 
              is around 1000 tokens, which is what prompted us to choose 250 and 500. We utilize a diversity threshold of 7 for all R1 results, 
              and we report all results across the two sets we partitioned, as mentioned in Experiment details.
            </p>
            
                <Figure 
                  src="/images/papers/interruption-paper/average_bar.png"
                  alt="Accuracy vs Refusal Rate Comparison"
                  title=""
                  caption=""
                  width="90%" 
                  height="auto"
                />
                
                <Figure 
                  src="/images/papers/interruption-paper/bar1.png"
                  alt=""
                  title=""
                  caption=""
                  width="90%" 
                  height="auto"
                />
                
                <Figure 
                  src="/images/papers/interruption-paper/bar2.png"
                  alt=""
                  title=""
                  caption=""
                  width="90%" 
                  height="auto"
                />
            
            <p>
              We find that we maintain accuracy on attempted questions close to our baseline models, while improving in refusal rates. 
              That is, while we are unable to add new knowledge to the model, we are able to leverage inference-time compute to accurately 
              judge when a model may be wrong.
            </p>
            <p>
              You may observe that our results from the second set of questions have much higher refusal rates— this is because the token 
              limits of 250 and 500 prevented several answers from completing, which were then parsed as NO_ATTEMPT by the SimpleQA grader. 
              We would like to introduce adaptive interrupt injection to solve this problem, but we decided not to pursue this given the 
              time constraints of the hackathon.
            </p>
      
          </PaperSection>
          
          <PaperSection id="conclusion" title="Conclusion">
            <p>
              Our work demonstrates that measuring diversity in parallel reasoning chains can serve as an effective signal for when reasoning 
              models should refuse to answer questions they would otherwise hallucinate on. By injecting an interruption token ("No, but") 
              at varying points in the reasoning process and analyzing the diversity of resulting thought patterns, we were able to maintain 
              accuracy on attempted questions while significantly improving refusal rates on potentially incorrect answers.
            </p>
            <p>
              This approach requires no additional model training and leverages inference-time computation to enable more trustworthy AI systems. 
              The method shows particular promise for high-stakes applications in medicine, law, and other domains where false information can 
              have serious consequences. Future work should focus on optimizing the interruption point through adaptive injection techniques, 
              exploring alternative interruption tokens, and extending this approach to other reasoning models beyond Deepseek R1.
            </p>
            <p>
              Overall, our method provides a practical framework for improving AI reliability through enhanced self-awareness of knowledge limitations.
            </p>
          </PaperSection>
          
          <PaperSection id="appendix" title="Appendix">
            <p>
              In our initial experiments with embedding-based diversity measurement, we found that semantic similarity metrics often 
              failed to capture the nuanced differences in reasoning chains, especially when those differences were concentrated in 
              relatively few tokens of long reasoning sequences.
            </p>
            
            <Figure 
              src="/images/papers/interruption-paper/matrix500.png"
              alt="Embedding similarity matrix"
              title="Figure 4: Embedding Similarity Matrix"
              caption="Cosine similarity between embeddings of different reasoning chains, showing high similarity despite meaningful reasoning differences."
              width="80%" 
              height="auto"
            />
            
            <p>
              We also experimented with different interruption tokens beyond "No, but", including tokens like "Wait", "Actually", and 
              simple punctuation marks like periods. In general, we found that tokens that explicitly introduce doubt or contradiction 
              (like "No, but") were most effective at generating diverse reasoning paths.
            </p>
            
            <p>
              Additional data and code implementation details are available in our GitHub repository.
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