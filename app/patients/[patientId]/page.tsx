'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { patientApi, documentApi } from '@/lib/api';
import { DocumentUploadDialog } from '@/components/patients/document-upload-dialog';
import Link from 'next/link';
import {
  Search,
  Upload,
  Star,
  MoreHorizontal,
  FileText,
  Calendar,
  Mail,
  Phone,
  MapPin,
  ArrowLeft,
  User,
  Activity,
  Clock
} from 'lucide-react';
import { format } from 'date-fns';

export default function PatientProfilePage() {
  const params = useParams();
  const patientId = params.patientId as string;
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

  const { data: patient, isLoading: patientLoading } = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => patientApi.getPatient(patientId),
  });

  const { data: documents, isLoading: documentsLoading, refetch: refetchDocuments } = useQuery({
    queryKey: ['patient-documents', patientId, activeFilter],
    queryFn: () => documentApi.getPatientDocuments(patientId, activeFilter),
  });

  const handleSearch = () => {
    if (searchQuery.trim()) {
      window.location.href = `/patients/${patientId}/search-results?q=${encodeURIComponent(searchQuery)}`;
    }
  };

  const handleUploadSuccess = () => {
    refetchDocuments();
    setUploadDialogOpen(false);
  };

  if (patientLoading) {
    return (
      <DashboardLayout>
        <div className="p-6 flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-gray-600">Loading patient information...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  if (!patient) {
    return (
      <DashboardLayout>
        <div className="p-6">
          <div className="text-center py-12">
            <User className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Patient Not Found</h2>
            <p className="text-gray-600">The patient you're looking for doesn't exist.</p>
            <Link href="/patients">
              <Button className="mt-4">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Patients
              </Button>
            </Link>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Breadcrumb */}
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <span>Pages</span>
          <span>/</span>
          <Link href="/patients" className="hover:text-primary">Patients</Link>
          <span>/</span>
          <span className="text-primary font-medium">Profile</span>
        </div>

        {/* Header */}
        <div className="flex items-center space-x-4">
          <Link href="/patients">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Patients
            </Button>
          </Link>
        </div>

        {/* Patient Info Card */}
        <Card className="bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center">
                  <span className="text-2xl font-semibold text-primary">
                    {patient.name.split(' ').map(n => n[0]).join('')}
                  </span>
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">{patient.name}</h1>
                  <div className="grid md:grid-cols-2 gap-4 mt-3 text-sm text-gray-600">
                    <div className="flex items-center space-x-2">
                      <Mail className="w-4 h-4" />
                      <span>{patient.email}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Phone className="w-4 h-4" />
                      <span>{patient.phone}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4" />
                      <span>Last visit: {format(new Date(patient.lastVisit), 'MMM dd, yyyy')}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <User className="w-4 h-4" />
                      <span>ID: {patient.enrollNumber}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end space-y-2">
                <Badge className="bg-green-100 text-green-800 border-green-200">Active Patient</Badge>
                <div className="text-right text-sm text-gray-500">
                  <p>Member since</p>
                  <p className="font-medium">{format(new Date(patient.createdAt), 'MMM yyyy')}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Search and Actions */}
        <Card>
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center space-y-4 md:space-y-0 md:space-x-4">
              <div className="flex-1 relative w-full">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Q Search through this patient's medical documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  className="pl-10"
                />
              </div>
              <div className="flex space-x-2">
                <Button onClick={handleSearch} disabled={!searchQuery.trim()}>
                  <Search className="w-4 h-4 mr-2" />
                  Search
                </Button>
                <Button
                  onClick={() => setUploadDialogOpen(true)}
                  className="medical-gradient text-white"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Import a doc
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Documents Section */}
        <Card>
          <CardHeader>
            <CardTitle>Medical Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeFilter} onValueChange={setActiveFilter} className="space-y-4">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="favorite" className="flex items-center space-x-2">
                  <Star className="w-4 h-4" />
                  <span>Favorite</span>
                </TabsTrigger>
                <TabsTrigger value="recent" className="flex items-center space-x-2">
                  <Clock className="w-4 h-4" />
                  <span>Recently added</span>
                </TabsTrigger>
                <TabsTrigger value="all" className="flex items-center space-x-2">
                  <FileText className="w-4 h-4" />
                  <span>All docs</span>
                </TabsTrigger>
              </TabsList>

              <TabsContent value={activeFilter} className="space-y-4">
                {documentsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                      <p className="text-gray-600">Loading documents...</p>
                    </div>
                  </div>
                ) : documents && documents.length > 0 ? (
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {documents.map((document) => (
                      <Card key={document.id} className="hover:shadow-lg transition-all duration-200 cursor-pointer group">
                        <Link href={`/patients/${patientId}/document/${document.id}`}>
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center space-x-2 flex-1">
                                <FileText className="w-5 h-5 text-primary flex-shrink-0" />
                                <span className="text-sm font-medium text-gray-900 truncate group-hover:text-primary transition-colors">
                                  {document.title}
                                </span>
                              </div>
                              <div className="flex items-center space-x-1 flex-shrink-0">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className={`p-1 ${document.isFavorite ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                                  onClick={(e) => {
                                    e.preventDefault();
                                    // Handle favorite toggle
                                  }}
                                >
                                  <Star className={`w-4 h-4 ${document.isFavorite ? 'fill-current' : ''}`} />
                                </Button>
                                <Button variant="ghost" size="sm" className="p-1 text-gray-400 hover:text-gray-600">
                                  <MoreHorizontal className="w-4 h-4" />
                                </Button>
                              </div>
                            </div>
                            
                            <p className="text-sm text-gray-600 mb-3 line-clamp-3 leading-relaxed">
                              {document.summary}
                            </p>
                            
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <div className="flex items-center space-x-1">
                                <Calendar className="w-3 h-3" />
                                <span>{format(new Date(document.uploadDate), 'MMM dd, yyyy')}</span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className="capitalize bg-gray-100 px-2 py-1 rounded text-xs">
                                  {document.fileType}
                                </span>
                                {document.fileSize && (
                                  <span className="text-gray-400">
                                    {(document.fileSize / (1024 * 1024)).toFixed(1)}MB
                                  </span>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Link>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
                    <p className="text-gray-600 mb-6 max-w-md mx-auto">
                      {activeFilter === 'favorite' 
                        ? "No favorite documents yet. Star documents to add them to your favorites." 
                        : activeFilter === 'recent'
                        ? "No recent documents found. Upload new documents to see them here."
                        : "No medical documents uploaded for this patient yet. Get started by uploading the first document."}
                    </p>
                    <Button
                      onClick={() => setUploadDialogOpen(true)}
                      className="medical-gradient text-white"
                    >
                      <Upload className="w-4 h-4 mr-2" />
                      {documents?.length === 0 ? 'Upload First Document' : 'Upload Document'}
                    </Button>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Upload Dialog */}
        <DocumentUploadDialog
          open={uploadDialogOpen}
          onOpenChange={setUploadDialogOpen}
          patientId={patientId}
          onSuccess={handleUploadSuccess}
        />
      </div>
    </DashboardLayout>
  );
}