import { ReactNode } from 'react';

// This function is required for static export with dynamic routes
export async function generateStaticParams() {
  // Return the list of patient and document ID combinations that should be statically generated
  // These correspond to the mock data in the API
  const params = [];
  
  // For each patient, add their documents
  const patientIds = ['1', '2', '3', '4', '5'];
  const documentIds = ['1', '2', '3', '4', '5'];
  
  for (const patientId of patientIds) {
    for (const documentId of documentIds) {
      params.push({ patientId, documentId });
    }
  }
  
  return params;
}

interface DocumentLayoutProps {
  children: ReactNode;
}

export default function DocumentLayout({ children }: DocumentLayoutProps) {
  return <>{children}</>;
}