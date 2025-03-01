# Research Data Viewer

A Next.js application for visualizing and exploring research data.

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

## Features

- Dynamic data loading from API
- Responsive design
- Dataset exploration interface
- Detailed dataset views

## Project Structure

- `app/` - Next.js app directory
  - `page.tsx` - Home page showing all datasets
  - `layout.tsx` - Root layout with metadata
  - `globals.css` - Global CSS styles
  - `api/` - API routes
    - `data/` - Research data endpoints
  - `datasets/` - Dataset detail pages

## API Routes

- `/api/data` - Returns all research data categories and datasets

## Adding New Datasets

To add new datasets:

1. Add your dataset to the returned data in `app/api/data/route.ts`
2. The UI will automatically display the new dataset

## Deployment

Build the application for production:

```bash
npm run build
```

Start the production server:

```bash
npm start
```