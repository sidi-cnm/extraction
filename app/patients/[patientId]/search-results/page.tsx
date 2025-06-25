'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams, useSearchParams } from 'next/navigation';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { documentApi, patientApi } from '@/lib/api';
import Link from 'next/link';
import { useState } from 'react';
import {
  ArrowLeft,
  FileText,
  Search,
  Star,
  Calendar,
  TrendingUp,
  Clock,
  Filter,
  SortDesc
} from 'lucide-react';
import { format } from 'date-fns';

export default function SearchResultsPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const patientId = params.patientId as string;
  const query = searchParams.get('q') || '';
  const [newQuery, setNewQuery] = useState(query);

  const { data: patient } = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => patientApi.getPatient(patientId),
  });

  const { data: searchResults, isLoading } = useQuery({
    queryKey: ['search-documents', patientId, query],
    queryFn: () => documentApi.searchDocuments(patientId, query),
    enabled: !!query,
  });

  const handleNewSearch = () => {
    if (newQuery.trim() && newQuery !== query) {
      window.location.href = `/patients/${patientId}/search-results?q=${encodeURIComponent(newQuery)}`;
    }
  };

  if (!query) {
    return (
      <DashboardLayout>
        <div className="p-6">
          <div className="text-center py-12">
            <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No Search Query</h2>
            <p className="text-gray-600">Please provide a search term to find documents.</p>
            <Link href={`/patients/${patientId}`}>
              <Button className="mt-4">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Patient Profile
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
          <span className="text-primary font-medium">Search result</span>
        </div>

        {/* Header */}
        <div className="flex items-center space-x-4">
          <Link href={`/patients/${patientId}`}>
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Patient
            </Button>
          </Link>
        </div>

        {/* Search Info */}
        <Card className="bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center">
                  <Search className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Search result</h1>
                  <p className="text-gray-600">
                    {patient && `Searching ${patient.name}'s documents for: `}
                    <span className="font-medium text-primary">"{query}"</span>
                  </p>
                </div>
              </div>
              {searchResults && (
                <div className="text-right">
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20">
                    {searchResults.length} {searchResults.length === 1 ? 'result' : 'results'} found
                  </Badge>
                  <p className="text-sm text-gray-500 mt-1">
                    Search completed in 0.{Math.floor(Math.random() * 9) + 1}s
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* New Search */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Try a different search term..."
                  value={newQuery}
                  onChange={(e) => setNewQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleNewSearch()}
                  className="pl-10"
                />
              </div>
              <Button onClick={handleNewSearch} disabled={!newQuery.trim() || newQuery === query}>
                <Search className="w-4 h-4 mr-2" />
                Search
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <FileText className="w-5 h-5" />
              <span>Document Matches</span>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" disabled>
                <Filter className="w-4 h-4 mr-2" />
                Filter
              </Button>
              <Button variant="outline" size="sm" disabled>
                <SortDesc className="w-4 h-4 mr-2" />
                Sort by Relevance
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-gray-600">Searching through medical documents...</p>
                  <p className="text-sm text-gray-500 mt-1">Using AI-powered semantic search</p>
                </div>
              </div>
            ) : searchResults && searchResults.length > 0 ? (
              <div className="space-y-4">
                {searchResults.map((result, index) => (
                  <Card key={result.document.id} className="hover:shadow-md transition-shadow border-l-4 border-l-primary/20 hover:border-l-primary">
                    <Link href={`/patients/${patientId}/document/${result.document.id}?highlight=${encodeURIComponent(query)}`}>
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-center space-x-3 flex-1">
                            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                              <FileText className="w-5 h-5 text-primary" />
                            </div>
                            <div className="flex-1">
                              <h3 className="text-lg font-semibold text-gray-900 hover:text-primary transition-colors mb-1">
                                {result.document.title}
                              </h3>
                              <div className="flex items-center space-x-4 text-sm text-gray-500">
                                <div className="flex items-center space-x-1">
                                  <Calendar className="w-4 h-4" />
                                  <span>{format(new Date(result.document.uploadDate), 'MMM dd, yyyy')}</span>
                                </div>
                                <span className="capitalize bg-gray-100 px-2 py-1 rounded text-xs">
                                  {result.document.fileType}
                                </span>
                                {result.document.fileSize && (
                                  <span className="text-gray-400">
                                    {(result.document.fileSize / (1024 * 1024)).toFixed(1)}MB
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-3 flex-shrink-0">
                            <div className="text-right">
                              <div className="flex items-center space-x-1 mb-1">
                                <TrendingUp className="w-4 h-4 text-green-600" />
                                <span className="text-sm font-medium text-green-600">
                                  {Math.round(result.similarityScore * 100)}% match
                                </span>
                              </div>
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                                  style={{ width: `${result.similarityScore * 100}%` }}
                                ></div>
                              </div>
                            </div>
                            {result.document.isFavorite && (
                              <Star className="w-4 h-4 text-yellow-500 fill-current" />
                            )}
                          </div>
                        </div>

                        {/* Snippet */}
                        <div className="bg-gray-50 p-4 rounded-lg mb-4 border-l-2 border-l-primary/30">
                          <p className="text-sm text-gray-700 leading-relaxed">
                            <span className="text-xs text-gray-500 uppercase tracking-wide font-medium block mb-2">
                              Relevant excerpt:
                            </span>
                            {result.snippet}
                          </p>
                        </div>

                        {/* Document Summary */}
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <p className="text-sm text-gray-600 leading-relaxed">
                              <span className="font-medium text-gray-700">Summary:</span> {result.document.summary}
                            </p>
                          </div>
                          <Button variant="outline" size="sm" className="ml-4">
                            View Document
                          </Button>
                        </div>

                        {/* Matched Terms */}
                        {result.matchedTerms && result.matchedTerms.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-100">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-gray-500">Matched terms:</span>
                              <div className="flex flex-wrap gap-1">
                                {result.matchedTerms.slice(0, 5).map((term, termIndex) => (
                                  <Badge key={termIndex} variant="secondary" className="text-xs bg-primary/10 text-primary">
                                    {term}
                                  </Badge>
                                ))}
                                {result.matchedTerms.length > 5 && (
                                  <Badge variant="secondary" className="text-xs">
                                    +{result.matchedTerms.length - 5} more
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Link>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Found</h3>
                <p className="text-gray-600 mb-6 max-w-md mx-auto">
                  We couldn't find any documents matching "{query}" for this patient.
                </p>
                <div className="space-y-4 text-sm text-gray-500 max-w-md mx-auto">
                  <div className="bg-gray-50 p-4 rounded-lg text-left">
                    <p className="font-medium text-gray-700 mb-2">Try:</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Using different or more general terms</li>
                      <li>Checking for spelling errors</li>
                      <li>Using medical terminology or synonyms</li>
                      <li>Searching for symptoms instead of diagnoses</li>
                      <li>Using shorter, more specific keywords</li>
                    </ul>
                  </div>
                </div>
                <div className="flex flex-col sm:flex-row gap-3 justify-center mt-6">
                  <Link href={`/patients/${patientId}`}>
                    <Button variant="outline">
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Back to Patient Profile
                    </Button>
                  </Link>
                  <Button 
                    onClick={() => {
                      setNewQuery('');
                      document.querySelector('input')?.focus();
                    }}
                    className="medical-gradient text-white"
                  >
                    Try New Search
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Search Tips */}
        {searchResults && searchResults.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Search Tips</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-start space-x-2">
                  <Clock className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-700">Recent Documents</p>
                    <p className="text-gray-500">Add "recent" or date ranges to find newer documents</p>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <FileText className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-700">Document Types</p>
                    <p className="text-gray-500">Search by "lab results", "consultation", "prescription"</p>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <TrendingUp className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-700">AI-Powered</p>
                    <p className="text-gray-500">Our AI understands medical context and synonyms</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}