import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Research Data Viewer',
  description: 'A dynamic website to display research data',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}