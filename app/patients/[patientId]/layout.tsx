import { ReactNode } from 'react';

// This function is required for static export with dynamic routes
export async function generateStaticParams() {
  // Return the list of patient IDs that should be statically generated
  // These correspond to the mock patient data in the API
  return [
    { patientId: '1' },
    { patientId: '2' },
    { patientId: '3' },
    { patientId: '4' },
    { patientId: '5' }
  ];
}

interface PatientLayoutProps {
  children: ReactNode;
}

export default function PatientLayout({ children }: PatientLayoutProps) {
  return <>{children}</>;
}