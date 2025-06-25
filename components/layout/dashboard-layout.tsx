'use client';

import { Sidebar } from './sidebar';
import { AuthGuard } from '@/components/auth/auth-guard';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <AuthGuard>
      <div className="flex h-screen bg-gray-50">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <div className="h-full medical-pattern">
            {children}
          </div>
        </main>
      </div>
    </AuthGuard>
  );
}