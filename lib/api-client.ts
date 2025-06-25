// Centralized API client with authentication and error handling
// This handles all HTTP requests and token management

import { User, Patient, Document, SearchResult, DocumentUploadResponse, PaginatedResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8001';

// Token management utilities
export const tokenManager = {
  getToken: (): string | null => {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('auth-token');
  },
  
  setToken: (token: string): void => {
    if (typeof window === 'undefined') return;
    localStorage.setItem('auth-token', token);
  },
  
  removeToken: (): void => {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('auth-token');
    localStorage.removeItem('user-data');
  },
  
  getUser: (): User | null => {
    if (typeof window === 'undefined') return null;
    const userData = localStorage.getItem('user-data');
    return userData ? JSON.parse(userData) : null;
  },
  
  setUser: (user: User): void => {
    if (typeof window === 'undefined') return;
    localStorage.setItem('user-data', JSON.stringify(user));
  }
};

// Generic API request function with proper error handling and authentication
async function apiRequest<T>(
  endpoint: string, 
  options: RequestInit = {},
  useFastAPI = false
): Promise<T> {
  const baseUrl = useFastAPI ? FASTAPI_BASE_URL : API_BASE_URL;
  const token = tokenManager.getToken();
  
  const config: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error(`API Request failed for ${endpoint}:`, error);
    throw error;
  }
}

// Authentication API
export const authApi = {
  async login(credentials: { email: string; password: string; rememberMe?: boolean }): Promise<{ user: User; token: string }> {
    // TODO: Integrate with real API endpoint: POST /api/auth/login
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
    
    // Mock successful login for demo credentials
    if (credentials.email === 'doctor@sanamed.com' && credentials.password === 'password') {
      const user: User = {
        id: '1',
        name: 'Dr. Sarah Johnson',
        email: credentials.email,
        role: 'doctor',
        createdAt: '2023-01-01T00:00:00Z'
      };
      const token = 'mock-jwt-token-' + Date.now();
      
      return { user, token };
    }
    
    throw new Error('Invalid credentials');
  },

  async signup(data: { name: string; email: string; password: string }): Promise<{ user: User; token: string }> {
    // TODO: Integrate with real API endpoint: POST /api/auth/signup
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const user: User = {
      id: Date.now().toString(),
      name: data.name,
      email: data.email,
      role: 'doctor',
      createdAt: new Date().toISOString()
    };
    const token = 'mock-jwt-token-' + Date.now();
    
    return { user, token };
  },

  async logout(): Promise<void> {
    // TODO: Integrate with real API endpoint: POST /api/auth/logout
    tokenManager.removeToken();
  }
};

// Patient API
export const patientApi = {
  async getPatients(page = 1, limit = 10, search = ''): Promise<PaginatedResponse<Patient>> {
    // TODO: Integrate with real API endpoint: GET /api/patients
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock patient data
    const mockPatients: Patient[] = [
      {
        id: '1',
        name: 'John Smith',
        email: 'john.smith@email.com',
        phone: '+1 (555) 123-4567',
        enrollNumber: 'EN001',
        lastVisit: '2024-01-15',
        createdAt: '2023-06-15',
        dateOfBirth: '1978-03-22',
        address: '123 Oak Street, Springfield, IL',
        medicalConditions: ['Hypertension', 'Type 2 Diabetes']
      },
      {
        id: '2',
        name: 'Emily Johnson',
        email: 'emily.johnson@email.com',
        phone: '+1 (555) 234-5678',
        enrollNumber: 'EN002',
        lastVisit: '2024-01-12',
        createdAt: '2023-07-20',
        dateOfBirth: '1985-07-14',
        address: '456 Pine Avenue, Springfield, IL',
        medicalConditions: ['Asthma']
      },
      {
        id: '3',
        name: 'Michael Brown',
        email: 'michael.brown@email.com',
        phone: '+1 (555) 345-6789',
        enrollNumber: 'EN003',
        lastVisit: '2024-01-10',
        createdAt: '2023-08-10',
        dateOfBirth: '1992-11-08',
        address: '789 Elm Drive, Springfield, IL',
        medicalConditions: []
      },
      {
        id: '4',
        name: 'Sarah Davis',
        email: 'sarah.davis@email.com',
        phone: '+1 (555) 456-7890',
        enrollNumber: 'EN004',
        lastVisit: '2024-01-08',
        createdAt: '2023-09-05',
        dateOfBirth: '1990-05-16',
        address: '321 Maple Lane, Springfield, IL',
        medicalConditions: ['Allergies', 'Migraine']
      },
      {
        id: '5',
        name: 'Robert Wilson',
        email: 'robert.wilson@email.com',
        phone: '+1 (555) 567-8901',
        enrollNumber: 'EN005',
        lastVisit: '2024-01-05',
        createdAt: '2023-10-12',
        dateOfBirth: '1965-12-03',
        address: '654 Cedar Court, Springfield, IL',
        medicalConditions: ['High Cholesterol', 'Arthritis']
      }
    ];

    // Filter patients based on search query
    const filtered = search 
      ? mockPatients.filter(p => 
          p.name.toLowerCase().includes(search.toLowerCase()) ||
          p.email.toLowerCase().includes(search.toLowerCase()) ||
          p.enrollNumber.toLowerCase().includes(search.toLowerCase())
        )
      : mockPatients;

    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedResults = filtered.slice(startIndex, endIndex);

    return {
      data: paginatedResults,
      total: filtered.length,
      page,
      totalPages: Math.ceil(filtered.length / limit),
      hasNext: page < Math.ceil(filtered.length / limit),
      hasPrev: page > 1
    };
  },

  async getPatient(id: string): Promise<Patient> {
    // TODO: Integrate with real API endpoint: GET /api/patients/{id}
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const mockPatients: Record<string, Patient> = {
      '1': {
        id: '1',
        name: 'John Smith',
        email: 'john.smith@email.com',
        phone: '+1 (555) 123-4567',
        enrollNumber: 'EN001',
        lastVisit: '2024-01-15',
        createdAt: '2023-06-15',
        dateOfBirth: '1978-03-22',
        address: '123 Oak Street, Springfield, IL 62701',
        emergencyContact: 'Jane Smith - +1 (555) 123-4568',
        medicalConditions: ['Hypertension', 'Type 2 Diabetes', 'High Cholesterol']
      }
    };

    return mockPatients[id] || mockPatients['1'];
  },

  async createPatient(patientData: Omit<Patient, 'id' | 'createdAt'>): Promise<Patient> {
    // TODO: Integrate with real API endpoint: POST /api/patients
    await new Promise(resolve => setTimeout(resolve, 800));
    
    return {
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      ...patientData
    };
  }
};

// Document API (FastAPI endpoints)
export const documentApi = {
  async getPatientDocuments(patientId: string, filter = 'all'): Promise<Document[]> {
    // TODO: Integrate with real FastAPI endpoint: GET /api/patients/{patientId}/documents
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const mockDocuments: Document[] = [
      {
        id: '1',
        patientId,
        title: 'Annual Physical Examination Report',
        content: 'Comprehensive annual physical examination completed...',
        summary: 'Routine physical exam - patient in good health, vitals normal, recommended follow-up in 12 months.',
        uploadDate: '2024-01-15',
        fileType: 'pdf',
        fileName: 'annual_physical_2024.pdf',
        fileSize: 2048576,
        isFavorite: true,
        tags: ['annual', 'physical', 'routine'],
        processingStatus: 'completed'
      },
      {
        id: '2',
        patientId,
        title: 'Blood Test Results - Complete Panel',
        content: 'Complete blood count and comprehensive metabolic panel results...',
        summary: 'Blood work shows normal values across all parameters. Cholesterol slightly elevated.',
        uploadDate: '2024-01-10',
        fileType: 'pdf',
        fileName: 'blood_test_jan2024.pdf',
        fileSize: 1024768,
        isFavorite: false,
        tags: ['blood test', 'lab results', 'cholesterol'],
        processingStatus: 'completed'
      }
    ];

    let filtered = mockDocuments;
    if (filter === 'favorite') {
      filtered = mockDocuments.filter(doc => doc.isFavorite);
    } else if (filter === 'recent') {
      filtered = mockDocuments.sort((a, b) => 
        new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime()
      ).slice(0, 10);
    }

    return filtered;
  },

  async searchDocuments(patientId: string, query: string): Promise<SearchResult[]> {
    // TODO: Integrate with real FastAPI endpoint: GET /api/patients/{patientId}/documents/search
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const mockResults: SearchResult[] = [
      {
        document: {
          id: '1',
          patientId,
          title: 'Annual Physical Examination Report',
          content: 'Patient presents for routine annual physical...',
          summary: 'Routine physical exam - patient in good health',
          uploadDate: '2024-01-15',
          fileType: 'pdf',
          fileName: 'annual_physical_2024.pdf',
          fileSize: 2048576,
          isFavorite: true,
          tags: ['annual', 'physical', 'routine'],
          processingStatus: 'completed'
        },
        snippet: `...Patient mentions occasional mild ${query} but attributes this to work stress...`,
        similarityScore: 0.94,
        matchedTerms: [query.toLowerCase(), 'symptoms', 'stress']
      }
    ];

    return mockResults;
  },

  async uploadDocument(patientId: string, file: File, title?: string): Promise<DocumentUploadResponse> {
    // TODO: Integrate with real FastAPI endpoint: POST /api/documents/upload
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const newDocument: Document = {
      id: Date.now().toString(),
      patientId,
      title: title || file.name.replace(/\.[^/.]+$/, ""),
      content: 'Document processing in progress...',
      summary: 'Document uploaded successfully. AI processing will extract key information.',
      uploadDate: new Date().toISOString().split('T')[0],
      fileType: file.type.split('/')[1] || 'pdf',
      fileName: file.name,
      fileSize: file.size,
      isFavorite: false,
      tags: [],
      processingStatus: 'processing'
    };

    return {
      id: newDocument.id,
      status: 'uploaded',
      message: 'Document uploaded successfully and is being processed',
      document: newDocument
    };
  },

  async getDocument(documentId: string): Promise<Document> {
    // TODO: Integrate with real FastAPI endpoint: GET /api/documents/{documentId}
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return {
      id: documentId,
      patientId: '1',
      title: 'Annual Physical Examination Report',
      content: `PATIENT: John Smith
DATE OF SERVICE: January 15, 2024
PROVIDER: Dr. Sarah Johnson, MD

CHIEF COMPLAINT:
Annual physical examination, no acute complaints.

HISTORY OF PRESENT ILLNESS:
Patient is a 45-year-old male presenting for routine annual physical examination. He reports feeling well overall with no specific concerns...

PHYSICAL EXAMINATION:
Vital Signs: 
- Blood Pressure: 128/82 mmHg
- Heart Rate: 72 bpm, regular
- Temperature: 98.6°F (37.0°C)
- Weight: 185 lbs (84 kg)
- Height: 5'10" (178 cm)
- BMI: 26.5

General: Well-appearing, alert and oriented, no acute distress...

ASSESSMENT AND PLAN:
1. HYPERTENSION - Well controlled
2. TYPE 2 DIABETES MELLITUS - Adequate control
3. HYPERLIPIDEMIA - Suboptimal control...`,
      summary: 'Annual physical examination completed. Patient in overall good health with well-controlled hypertension.',
      uploadDate: '2024-01-15',
      fileType: 'pdf',
      fileName: 'annual_physical_2024.pdf',
      fileSize: 2048576,
      isFavorite: true,
      tags: ['annual', 'physical', 'routine'],
      processingStatus: 'completed'
    };
  }
};