// Patient list page with search, pagination, and add patient functionality
// Includes comprehensive table layout and responsive design

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { AddPatientModal } from '@/components/patients/add-patient-modal';
import { patientApi } from '@/lib/api-client';
import Link from 'next/link';
import {
  Search,
  Plus,
  MoreHorizontal,
  Edit,
  Trash2,
  Eye,
  ChevronLeft,
  ChevronRight,
  Users,
  FileText,
  Calendar
} from 'lucide-react';
import { format } from 'date-fns';

export default function PatientsPage() {
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [addPatientModalOpen, setAddPatientModalOpen] = useState(false);
  const limit = 10;

  // Fetch patients with React Query for caching and revalidation
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['patients', page, search],
    queryFn: () => patientApi.getPatients(page, limit, search),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const patients = data?.data || [];
  const totalPages = data?.totalPages || 1;
  const total = data?.total || 0;

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1); // Reset to first page when searching
    refetch();
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2 text-sm text-gray-500 mb-2">
              <span>Pages</span>
              <span>/</span>
              <span className="text-primary font-medium">Patients</span>
            </div>
            <h1 className="text-3xl font-bold text-gray-900">Patient page</h1>
            <p className="text-gray-600 mt-2">Manage your patient records and medical history</p>
          </div>
          <Button 
            className="medical-gradient text-white"
            onClick={() => setAddPatientModalOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            New Patient
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Users className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total Patients</p>
                  <p className="text-2xl font-bold text-gray-900">{total}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Documents</p>
                  <p className="text-2xl font-bold text-gray-900">1,429</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Calendar className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">This Month</p>
                  <p className="text-2xl font-bold text-gray-900">24</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Search and Table */}
        <Card>
          <CardHeader>
            <CardTitle>Patients Table</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Search */}
            <form onSubmit={handleSearch} className="flex items-center space-x-4">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search patients by name, email, or enroll number..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Button type="submit" disabled={isLoading}>
                <Search className="w-4 h-4 mr-2" />
                Search
              </Button>
            </form>

            {/* Error Display */}
            {error && (
              <ErrorMessage 
                message={error instanceof Error ? error.message : 'Failed to load patients'} 
              />
            )}

            {/* Table */}
            <div className="rounded-lg border bg-white">
              <Table>
                <TableHeader>
                  <TableRow className="bg-gray-50">
                    <TableHead className="font-semibold">Name</TableHead>
                    <TableHead className="font-semibold">Email</TableHead>
                    <TableHead className="font-semibold">Phone</TableHead>
                    <TableHead className="font-semibold">Enroll Number</TableHead>
                    <TableHead className="font-semibold">Last Visit</TableHead>
                    <TableHead className="font-semibold">Status</TableHead>
                    <TableHead className="text-right font-semibold">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center py-12">
                        <div className="flex flex-col items-center space-y-3">
                          <LoadingSpinner />
                          <p className="text-gray-500">Loading patients...</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : patients.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center py-12">
                        <div className="flex flex-col items-center space-y-3">
                          <Users className="w-12 h-12 text-gray-400" />
                          <p className="text-gray-500 font-medium">
                            {search ? 'No patients found matching your search' : 'No patients found'}
                          </p>
                          {search && (
                            <Button 
                              variant="outline" 
                              onClick={() => {
                                setSearch('');
                                setPage(1);
                              }}
                            >
                              Clear search
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : (
                    patients.map((patient) => (
                      <TableRow key={patient.id} className="hover:bg-gray-50">
                        <TableCell className="font-medium">
                          <Link 
                            href={`/patients/${patient.id}`}
                            className="text-primary hover:underline font-semibold"
                          >
                            {patient.name}
                          </Link>
                        </TableCell>
                        <TableCell className="text-gray-600">{patient.email}</TableCell>
                        <TableCell className="text-gray-600">{patient.phone}</TableCell>
                        <TableCell className="text-gray-600 font-medium">{patient.enrollNumber}</TableCell>
                        <TableCell className="text-gray-600">
                          {format(new Date(patient.lastVisit), 'MMM dd, yyyy')}
                        </TableCell>
                        <TableCell>
                          <Badge 
                            variant="secondary" 
                            className="bg-green-100 text-green-800 border-green-200"
                          >
                            Active
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end space-x-1">
                            <Link href={`/patients/${patient.id}`}>
                              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                <Eye className="w-4 h-4" />
                              </Button>
                            </Link>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                              <Edit className="w-4 h-4" />
                            </Button>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-red-600 hover:text-red-700 hover:bg-red-50">
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-4">
                <p className="text-sm text-gray-700">
                  Showing page {page} of {totalPages} ({total} total patients)
                </p>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(Math.max(1, page - 1))}
                    disabled={page === 1 || isLoading}
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </Button>
                  
                  {/* Page numbers */}
                  <div className="flex items-center space-x-1">
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      const pageNum = i + 1;
                      return (
                        <Button
                          key={pageNum}
                          variant={page === pageNum ? "default" : "outline"}
                          size="sm"
                          onClick={() => handlePageChange(pageNum)}
                          disabled={isLoading}
                          className="w-8 h-8 p-0"
                        >
                          {pageNum}
                        </Button>
                      );
                    })}
                    {totalPages > 5 && (
                      <>
                        <span className="text-gray-500">...</span>
                        <Button
                          variant={page === totalPages ? "default" : "outline"}
                          size="sm"
                          onClick={() => handlePageChange(totalPages)}
                          disabled={isLoading}
                          className="w-8 h-8 p-0"
                        >
                          {totalPages}
                        </Button>
                      </>
                    )}
                  </div>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(Math.min(totalPages, page + 1))}
                    disabled={page === totalPages || isLoading}
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Add Patient Modal */}
        <AddPatientModal
          open={addPatientModalOpen}
          onOpenChange={setAddPatientModalOpen}
        />
      </div>
    </DashboardLayout>
  );
}