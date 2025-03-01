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
            description: 'Evaluation of question answering responses',
            sampleSize: 500
          },
          {
            id: 'simpleqa_continuations',
            title: 'Simple QA Continuations',
            description: 'Continuations of question answering tasks',
            sampleSize: 500
          },
          {
            id: 'consensus_responses',
            title: 'Consensus Responses',
            description: 'Aggregated consensus from multiple model responses',
            sampleSize: 250
          }
        ]
      }
    ]
  };

  return NextResponse.json(researchData);
}