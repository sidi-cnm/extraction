// Core TypeScript interfaces for the SanaMed application
// These define the data structures used throughout the app

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'doctor' | 'admin';
  createdAt: string;
}

export interface Patient {
  id: string;
  name: string;
  email: string;
  phone: string;
  enrollNumber: string;
  lastVisit: string;
  createdAt: string;
  dateOfBirth?: string;
  address?: string;
  emergencyContact?: string;
  medicalConditions?: string[];
}

export interface Document {
  id: string;
  patientId: string;
  title: string;
  content: string;
  summary: string;
  uploadDate: string;
  fileType: string;
  fileName: string;
  fileSize?: number;
  isFavorite: boolean;
  similarityScore?: number;
  tags?: string[];
  processingStatus?: 'pending' | 'processing' | 'completed' | 'failed';
}

export interface SearchResult {
  document: Document;
  snippet: string;
  similarityScore: number;
  matchedTerms: string[];
  highlightedContent?: string;
}

export interface DocumentUploadResponse {
  id: string;
  status: 'uploaded' | 'processing' | 'completed';
  message: string;
  document?: Document;
}

// API Response types
export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Form types
export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface SignupData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export interface CreatePatientData {
  name: string;
  email: string;
  phone: string;
  enrollNumber: string;
  dateOfBirth?: string;
  address?: string;
  emergencyContact?: string;
}