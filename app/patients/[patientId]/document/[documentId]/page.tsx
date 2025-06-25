'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams, useSearchParams } from 'next/navigation';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { documentApi, patientApi } from '@/lib/api';
import Link from 'next/link';
import {
  ArrowLeft,
  FileText,
  Download,
  Eye,
  Calendar,
  Star,
  Share,
  MoreHorizontal,
  User,
  Clock,
  FileType,
  Bookmark,
  Edit,
  Trash2
} from 'lucide-react';
import { format } from 'date-fns';

export default function DocumentViewPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const patientId = params.patientId as string;
  const documentId = params.documentId as string;
  const highlight = searchParams.get('highlight');

  const { data: patient } = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => patientApi.getPatient(patientId),
  });

  const { data: document, isLoading } = useQuery({
    queryKey: ['document', documentId],
    queryFn: () => documentApi.getDocument(documentId),
  });

  // Function to highlight search terms in text
  const highlightText = (text: string, searchTerm: string | null) => {
    if (!searchTerm) return text;
    
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <span key={index} className="bg-yellow-200 px-1 rounded font-medium">
          {part}
        </span>
      ) : part
    );
  };

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="p-6 flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-gray-600">Loading document...</p>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  if (!document) {
    return (
      <DashboardLayout>
        <div className="p-6">
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Document Not Found</h2>
            <p className="text-gray-600">The document you're looking for doesn't exist.</p>
            <Link href={`/patients/${patientId}`}>
              <Button className="mt-4">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Patient
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
          <Link href={`/patients/${patientId}`} className="hover:text-primary">Profile</Link>
          <span>/</span>
          <span className="text-primary font-medium">view document</span>
        </div>

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href={`/patients/${patientId}`}>
              <Button variant="ghost" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Patient
              </Button>
            </Link>
            {patient && (
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{document.title}</h1>
                <p className="text-gray-600">Patient: {patient.name}</p>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Eye className="w-4 h-4 mr-2" />
              View document
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
            <Button variant="outline" size="sm">
              <Share className="w-4 h-4 mr-2" />
              Share
            </Button>
            <Button variant="outline" size="sm">
              <MoreHorizontal className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Highlight Notice */}
        {highlight && (
          <Card className="bg-yellow-50 border-yellow-200">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Bookmark className="w-4 h-4 text-yellow-600" />
                <span className="text-sm text-yellow-800">
                  Highlighting search term: <strong>"{highlight}"</strong>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Document Layout */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Document Content */}
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Document Content</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-gray-50 p-6 rounded-lg max-h-[600px] overflow-y-auto">
                  <div className="whitespace-pre-wrap text-sm leading-relaxed text-gray-800 font-mono">
                    {highlightText(document.content, highlight)}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Document Metadata */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Document Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">File Name</label>
                  <p className="text-sm text-gray-900 mt-1 break-all">{document.fileName}</p>
                </div>
                
                <Separator />
                
                <div>
                  <label className="text-sm font-medium text-gray-500">Upload Date</label>
                  <div className="flex items-center space-x-2 mt-1">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-900">
                      {format(new Date(document.uploadDate), 'MMMM dd, yyyy')}
                    </span>
                  </div>
                </div>
                
                <Separator />
                
                <div>
                  <label className="text-sm font-medium text-gray-500">File Type</label>
                  <div className="flex items-center space-x-2 mt-1">
                    <FileType className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-900 capitalize">{document.fileType}</span>
                  </div>
                </div>

                {document.fileSize && (
                  <>
                    <Separator />
                    <div>
                      <label className="text-sm font-medium text-gray-500">File Size</label>
                      <p className="text-sm text-gray-900 mt-1">
                        {(document.fileSize / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </>
                )}

                <Separator />

                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <div className="flex items-center space-x-2 mt-1">
                    {document.isFavorite && (
                      <Star className="w-4 h-4 text-yellow-500 fill-current" />
                    )}
                    <Badge variant="secondary" className="bg-green-100 text-green-800">
                      {document.processingStatus === 'completed' ? 'Processed' : 'Processing'}
                    </Badge>
                  </div>
                </div>

                {document.similarityScore && (
                  <>
                    <Separator />
                    <div>
                      <label className="text-sm font-medium text-gray-500">Relevance Score</label>
                      <div className="mt-1">
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${document.similarityScore * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-green-600">
                            {Math.round(document.similarityScore * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {document.tags && document.tags.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <label className="text-sm font-medium text-gray-500">Tags</label>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {document.tags.map((tag, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>AI Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {document.summary}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Star className="w-4 h-4 mr-2" />
                  {document.isFavorite ? 'Remove from Favorites' : 'Add to Favorites'}
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Edit className="w-4 h-4 mr-2" />
                  Edit Document Info
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <FileText className="w-4 h-4 mr-2" />
                  Generate Report
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Download className="w-4 h-4 mr-2" />
                  Download Original
                </Button>
                <Separator />
                <Button variant="outline" className="w-full justify-start text-red-600 hover:text-red-700 hover:bg-red-50">
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Document
                </Button>
              </CardContent>
            </Card>

            {/* Processing Info */}
            {document.processingStatus === 'processing' && (
              <Card className="bg-blue-50 border-blue-200">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Clock className="w-4 h-4 text-blue-600 animate-spin" />
                    <div className="text-sm text-blue-800">
                      <p className="font-medium">AI Processing in Progress</p>
                      <p>Content analysis and summary generation in progress...</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}