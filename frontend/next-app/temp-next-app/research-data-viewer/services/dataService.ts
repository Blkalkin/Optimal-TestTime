/**
 * Service to handle data loading for the research data viewer
 */

export interface Dataset {
  id: string;
  title: string;
  description: string;
  sampleSize: number;
}

export interface Category {
  id: string;
  title: string;
  datasets: Dataset[];
}

export interface ResearchData {
  categories: Category[];
}

/**
 * Get all research data categories and datasets
 */
export const getResearchData = async (): Promise<ResearchData> => {
  try {
    const response = await fetch('/api/data');
    if (!response.ok) {
      throw new Error('Failed to fetch research data');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching research data:', error);
    throw error;
  }
};

/**
 * Get dataset details by category and dataset id
 * In a real app, this would fetch from the backend
 */
export const getDatasetDetails = async (categoryId: string, datasetId: string) => {
  try {
    // For now, this creates mock data based on the IDs
    // In a real app, you would fetch this from an API endpoint
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      title: `${datasetId} Dataset`.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      description: `This is the detailed view of the ${datasetId} dataset in the ${categoryId} category.`,
      sampleSize: Math.floor(Math.random() * 500) + 100,
      data: Array.from({ length: 10 }, (_, i) => ({
        id: i,
        question: `Sample question ${i + 1}?`,
        answer: `Sample answer ${i + 1} for the ${datasetId} dataset.`,
        score: Math.random().toFixed(2)
      }))
    };
  } catch (error) {
    console.error('Error fetching dataset details:', error);
    throw error;
  }
};