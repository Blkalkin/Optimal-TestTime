# Research Data Viewer

A modern, single-page application for visualizing and exploring research data on hallucinations and model performance.

## Features

- **Modern UI**: Sleek, responsive design with smooth scrolling sections
- **Interactive Data Exploration**: Click on datasets to view detailed information
- **Data Visualization**: Simple visual representations of research findings
- **Modular Structure**: Components-based architecture for easy extension
- **Responsive Design**: Works on all device sizes

## Getting Started

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

- `app/` - Next.js app directory
  - `page.tsx` - Main single-page application with all sections
  - `layout.tsx` - Root layout component
  - `globals.css` - All styles for the application
  - `api/` - API routes for data
- `services/` - Service layer for data fetching and processing
- `components/` - (Future) Reusable UI components

## Extending the Application

### Adding New Sections

1. Create a new section component in the `page.tsx` file:
```tsx
<Section id="new-section" title="Your New Section">
  <div className="your-section-content">
    {/* Your content here */}
  </div>
</Section>
```

2. Add a new navigation link in the `Navigation` component:
```tsx
<button onClick={() => scrollToSection('new-section')}>New Section</button>
```

### Adding New Datasets

1. Update the API route in `app/api/data/route.ts` to include your new dataset:
```ts
{
  id: 'your_dataset_id',
  title: 'Your Dataset Title',
  description: 'Description of your dataset',
  sampleSize: 300
}
```

### Adding New Visualizations

1. Add your visualization in the Visualizations section:
```tsx
<div className="visualization-card">
  <h3 className="viz-title">Your Visualization Title</h3>
  <div className="your-visualization">
    {/* Your visualization content */}
  </div>
  <p className="viz-description">
    Description of your visualization
  </p>
</div>
```

## Styling

The application uses CSS variables for consistent theming. You can modify the theme by updating the variables in `:root` at the top of `globals.css`.

## Future Enhancements

- Add real data integration with research datasets
- Implement more sophisticated data visualizations
- Add user authentication for restricted datasets
- Create a comparison tool for different research data
- Add search functionality for datasets

## License

Open source and free to use for research purposes.