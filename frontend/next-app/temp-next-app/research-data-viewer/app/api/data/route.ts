import { NextResponse } from 'next/server';

export async function GET() {
  // This is sample data. In a real app, you would fetch this from your database or files
  const researchData = {
    categories: [
      {
        id: 'hallucinations',
        title: 'Hallucinations',
        datasets: [
          {
            id: 'simpleqa_responses',
            title: 'Simple QA Responses',
            description: 'Evaluation of question answering responses for hallucination detection',
            sampleSize: 500
          },
          {
            id: 'simpleqa_continuations',
            title: 'Simple QA Continuations',
            description: 'Analysis of model behavior when continuing responses to questions',
            sampleSize: 500
          },
          {
            id: 'consensus_responses',
            title: 'Consensus Responses',
            description: 'Aggregated consensus from multiple model responses to evaluate agreement',
            sampleSize: 250
          }
        ]
      },
      {
        id: 'performance',
        title: 'Model Performance',
        datasets: [
          {
            id: 'accuracy_metrics',
            title: 'Accuracy Metrics',
            description: 'Comprehensive accuracy metrics across different task types',
            sampleSize: 1000
          },
          {
            id: 'response_time',
            title: 'Response Time Analysis',
            description: 'Evaluation of model response time under different conditions',
            sampleSize: 750
          }
        ]
      }
    ]
  };

  return NextResponse.json(researchData);
}