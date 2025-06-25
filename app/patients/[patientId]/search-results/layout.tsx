import { ReactNode } from 'react';

// This function is required for static export with dynamic routes
export async function generateStaticParams() {
  // Return the list of patient IDs that should be statically generated
  return [
    { patientId: '1' },
    { patientId: '2' },
    { patientId: '3' },
    { patientId: '4' },
    { patientId: '5' }
  ];
}

interface SearchResultsLayoutProps {
  children: ReactNode;
}

export default function SearchResultsLayout({ children }: SearchResultsLayoutProps) {
  return <>{children}</>;
}