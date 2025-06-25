import { authApi } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8001';

// Generic API request function with proper error handling
async function apiRequest<T>(
  endpoint: string, 
  options: RequestInit = {},
  useFastAPI = false
): Promise<T> {
  const baseUrl = useFastAPI ? FASTAPI_BASE_URL : API_BASE_URL;
  const token = authApi.getToken();
  
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

// Patient API types
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

// Patient API
export const patientApi = {
  async getPatients(page = 1, limit = 10, search = ''): Promise<{
    patients: Patient[];
    total: number;
    page: number;
    totalPages: number;
  }> {
    // Mock data - In production, this would call: await apiRequest('/api/patients', { method: 'GET' })
    await new Promise(resolve => setTimeout(resolve, 500));
    
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
      patients: paginatedResults,
      total: filtered.length,
      page,
      totalPages: Math.ceil(filtered.length / limit)
    };
  },

  async getPatient(id: string): Promise<Patient> {
    // Mock API call - In production: await apiRequest(`/api/patients/${id}`)
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
      },
      '2': {
        id: '2',
        name: 'Emily Johnson',
        email: 'emily.johnson@email.com',
        phone: '+1 (555) 234-5678',
        enrollNumber: 'EN002',
        lastVisit: '2024-01-12',
        createdAt: '2023-07-20',
        dateOfBirth: '1985-07-14',
        address: '456 Pine Avenue, Springfield, IL 62702',
        emergencyContact: 'Mark Johnson - +1 (555) 234-5679',
        medicalConditions: ['Asthma', 'Seasonal Allergies']
      }
    };

    const patient = mockPatients[id] || {
      id,
      name: 'John Smith',
      email: 'john.smith@email.com',
      phone: '+1 (555) 123-4567',
      enrollNumber: 'EN001',
      lastVisit: '2024-01-15',
      createdAt: '2023-06-15'
    };

    return patient;
  },

  async createPatient(patientData: Omit<Patient, 'id' | 'createdAt'>): Promise<Patient> {
    // Mock API call - In production: await apiRequest('/api/patients', { method: 'POST', body: JSON.stringify(patientData) })
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
    // FastAPI endpoint: GET /api/patients/{patientId}/documents
    // In production: await apiRequest(`/api/patients/${patientId}/documents?filter=${filter}`, {}, true)
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const mockDocuments: Document[] = [
      {
        id: '1',
        patientId,
        title: 'Annual Physical Examination Report',
        content: 'Comprehensive annual physical examination completed on January 15, 2024...',
        summary: 'Routine physical exam - patient in good health, vitals normal, recommended follow-up in 12 months.',
        uploadDate: '2024-01-15',
        fileType: 'pdf',
        fileName: 'annual_physical_2024.pdf',
        fileSize: 2048576, // 2MB
        isFavorite: true,
        tags: ['annual', 'physical', 'routine'],
        processingStatus: 'completed'
      },
      {
        id: '2',
        patientId,
        title: 'Blood Test Results - Complete Panel',
        content: 'Complete blood count and comprehensive metabolic panel results...',
        summary: 'Blood work shows normal values across all parameters. Cholesterol slightly elevated - recommend dietary modifications.',
        uploadDate: '2024-01-10',
        fileType: 'pdf',
        fileName: 'blood_test_jan2024.pdf',
        fileSize: 1024768, // 1MB
        isFavorite: false,
        tags: ['blood test', 'lab results', 'cholesterol'],
        processingStatus: 'completed'
      },
      {
        id: '3',
        patientId,
        title: 'Cardiology Consultation Report',
        content: 'Patient referred for cardiac evaluation due to chest pain episodes...',
        summary: 'Cardiology consult - stress test normal, EKG shows minor abnormalities, recommend lifestyle changes.',
        uploadDate: '2024-01-05',
        fileType: 'pdf',
        fileName: 'cardiology_consult.pdf',
        fileSize: 3145728, // 3MB
        isFavorite: false,
        tags: ['cardiology', 'consultation', 'chest pain'],
        processingStatus: 'completed'
      },
      {
        id: '4',
        patientId,
        title: 'Prescription Record - Diabetes Medication',
        content: 'Updated prescription for diabetes management including metformin dosage adjustment...',
        summary: 'Prescription updated for better diabetes control. Patient responding well to current regimen.',
        uploadDate: '2024-01-03',
        fileType: 'pdf',
        fileName: 'prescription_diabetes_jan2024.pdf',
        fileSize: 512000, // 512KB
        isFavorite: true,
        tags: ['prescription', 'diabetes', 'medication'],
        processingStatus: 'completed'
      },
      {
        id: '5',
        patientId,
        title: 'Radiology Report - Chest X-Ray',
        content: 'Chest X-ray performed due to persistent cough symptoms...',
        summary: 'Chest X-ray shows clear lungs, no signs of infection or abnormalities.',
        uploadDate: '2023-12-28',
        fileType: 'pdf',
        fileName: 'chest_xray_dec2023.pdf',
        fileSize: 4194304, // 4MB
        isFavorite: false,
        tags: ['radiology', 'x-ray', 'chest', 'cough'],
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

  async getDocument(documentId: string): Promise<Document> {
    // FastAPI endpoint: GET /api/documents/{documentId}
    // In production: await apiRequest(`/api/documents/${documentId}`, {}, true)
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const mockDocuments: Record<string, Document> = {
      '1': {
        id: '1',
        patientId: '1',
        title: 'Annual Physical Examination Report',
        content: `PATIENT: John Smith
DATE OF SERVICE: January 15, 2024
PROVIDER: Dr. Sarah Johnson, MD
FACILITY: Springfield Medical Center

CHIEF COMPLAINT:
Annual physical examination, no acute complaints.

HISTORY OF PRESENT ILLNESS:
Patient is a 45-year-old male presenting for routine annual physical examination. He reports feeling well overall with no specific concerns. He mentions occasional mild fatigue but attributes this to work stress and irregular sleep patterns. No chest pain, shortness of breath, or other cardiovascular symptoms. No gastrointestinal complaints. No urinary symptoms.

PAST MEDICAL HISTORY:
- Hypertension, diagnosed 2018, well-controlled on medication
- Hyperlipidemia, diagnosed 2019
- Type 2 Diabetes Mellitus, diagnosed 2020, managed with metformin
- No significant surgical history
- No known drug allergies

MEDICATIONS:
- Lisinopril 10mg daily for hypertension
- Atorvastatin 20mg daily for hyperlipidemia  
- Metformin 1000mg twice daily for diabetes
- Multivitamin daily

SOCIAL HISTORY:
- Married, works as accountant
- Non-smoker (quit 5 years ago, 20 pack-year history)
- Occasional alcohol use (1-2 drinks per week)
- Regular exercise 3x per week (walking, swimming)
- Diet: trying to follow diabetic diet recommendations

FAMILY HISTORY:
- Father: deceased at 70, myocardial infarction
- Mother: alive at 72, hypertension, osteoporosis
- No siblings
- No family history of cancer

REVIEW OF SYSTEMS:
Constitutional: Denies fever, chills, weight loss
Cardiovascular: Denies chest pain, palpitations, leg swelling
Respiratory: Denies cough, shortness of breath
Gastrointestinal: Denies nausea, vomiting, abdominal pain
Genitourinary: Denies dysuria, frequency
Neurological: Denies headaches, dizziness
Musculoskeletal: Mild lower back stiffness, no joint pain

PHYSICAL EXAMINATION:
Vital Signs: 
- Blood Pressure: 128/82 mmHg
- Heart Rate: 72 bpm, regular
- Temperature: 98.6°F (37.0°C)
- Weight: 185 lbs (84 kg)
- Height: 5'10" (178 cm)
- BMI: 26.5

General: Well-appearing, alert and oriented, no acute distress
HEENT: Normocephalic, atraumatic. Pupils equal, round, reactive to light. Extraocular movements intact. Tympanic membranes clear. Oropharynx clear.
Neck: Supple, no lymphadenopathy, no thyromegaly
Cardiovascular: Regular rate and rhythm, normal S1 and S2, no murmurs, rubs, or gallops
Pulmonary: Clear to auscultation bilaterally, no wheezes, rales, or rhonchi
Abdomen: Soft, non-tender, non-distended, bowel sounds present, no hepatosplenomegaly
Extremities: No cyanosis, clubbing, or edema. Pulses intact.
Skin: No rashes or lesions noted
Neurological: Alert and oriented x3, cranial nerves II-XII intact, strength 5/5 in all extremities

LABORATORY RESULTS (fasting):
- Hemoglobin A1c: 7.1% (target <7%)
- Fasting glucose: 142 mg/dL
- Total cholesterol: 198 mg/dL
- LDL cholesterol: 115 mg/dL
- HDL cholesterol: 42 mg/dL
- Triglycerides: 205 mg/dL
- Creatinine: 1.0 mg/dL
- eGFR: >60 mL/min/1.73m²

ASSESSMENT AND PLAN:

1. HYPERTENSION - Well controlled
   - Continue Lisinopril 10mg daily
   - Home blood pressure monitoring
   - Follow-up in 6 months

2. TYPE 2 DIABETES MELLITUS - Adequate control
   - A1c slightly above target at 7.1%
   - Continue Metformin 1000mg twice daily
   - Dietary counseling reinforced
   - Consider additional medication if A1c not improved in 3 months
   - Annual eye exam scheduled
   - Annual foot exam performed - no issues

3. HYPERLIPIDEMIA - Suboptimal control
   - LDL and triglycerides elevated
   - Increase Atorvastatin to 40mg daily
   - Dietary modifications discussed
   - Recheck lipid panel in 6 weeks

4. HEALTH MAINTENANCE
   - Influenza vaccination administered
   - Colonoscopy screening discussed (due at age 45)
   - Recommended screening colonoscopy within next 6 months
   - Prostate screening discussed - will consider PSA next year
   - Continue current exercise regimen
   - Stress management techniques discussed

5. FATIGUE
   - Likely related to work stress and sleep patterns
   - Sleep hygiene counseling provided
   - Recommend stress reduction techniques
   - Consider sleep study if symptoms persist

FOLLOW-UP:
- Return to clinic in 3 months for diabetes and lipid management
- Annual physical examination in 12 months
- Laboratory follow-up in 6 weeks (lipid panel)
- Contact office if any concerns or new symptoms develop

Patient counseled on all findings and agrees with treatment plan. All questions answered.

Dr. Sarah Johnson, MD
Internal Medicine
Springfield Medical Center`,
        summary: 'Annual physical examination completed. Patient in overall good health with well-controlled hypertension. Diabetes management adequate but A1c slightly elevated at 7.1%. Hyperlipidemia requires medication adjustment. Recommended colonoscopy screening and continued lifestyle modifications.',
        uploadDate: '2024-01-15',
        fileType: 'pdf',
        fileName: 'annual_physical_2024.pdf',
        fileSize: 2048576,
        isFavorite: true,
        tags: ['annual', 'physical', 'routine', 'diabetes', 'hypertension'],
        processingStatus: 'completed'
      }
    };

    return mockDocuments[documentId] || mockDocuments['1'];
  },

  async searchDocuments(patientId: string, query: string): Promise<SearchResult[]> {
    // FastAPI endpoint: POST /api/patients/{patientId}/documents/search
    // In production: await apiRequest(`/api/patients/${patientId}/documents/search`, { method: 'POST', body: JSON.stringify({ query }) }, true)
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Simulate AI-powered semantic search results
    const mockResults: SearchResult[] = [
      {
        document: {
          id: '1',
          patientId,
          title: 'Annual Physical Examination Report',
          content: 'Patient presents for routine annual physical...',
          summary: 'Routine physical exam - patient in good health with controlled conditions',
          uploadDate: '2024-01-15',
          fileType: 'pdf',
          fileName: 'annual_physical_2024.pdf',
          fileSize: 2048576,
          isFavorite: true,
          tags: ['annual', 'physical', 'routine'],
          processingStatus: 'completed'
        },
        snippet: `...Patient mentions occasional mild ${query} but attributes this to work stress and irregular sleep patterns. No chest pain, shortness of breath, or other cardiovascular symptoms...`,
        similarityScore: 0.94,
        matchedTerms: [query.toLowerCase(), 'symptoms', 'stress'],
        highlightedContent: `Patient mentions occasional mild <mark>${query}</mark> but attributes this to work stress`
      },
      {
        document: {
          id: '3',
          patientId,
          title: 'Cardiology Consultation Report',
          content: 'Patient referred for cardiac evaluation...',
          summary: 'Cardiology consult - stress test normal, EKG minor abnormalities',
          uploadDate: '2024-01-05',
          fileType: 'pdf',
          fileName: 'cardiology_consult.pdf',
          fileSize: 3145728,
          isFavorite: false,
          tags: ['cardiology', 'consultation', 'chest pain'],
          processingStatus: 'completed'
        },
        snippet: `...Stress test results show normal cardiac function. Patient experienced ${query} during initial evaluation but subsequent monitoring shows improvement...`,
        similarityScore: 0.87,
        matchedTerms: [query.toLowerCase(), 'cardiac', 'evaluation'],
        highlightedContent: `Patient experienced <mark>${query}</mark> during initial evaluation`
      },
      {
        document: {
          id: '2',
          patientId,
          title: 'Blood Test Results - Complete Panel',
          content: 'Complete blood count and metabolic panel results...',
          summary: 'Blood work normal, cholesterol slightly elevated',
          uploadDate: '2024-01-10',
          fileType: 'pdf',
          fileName: 'blood_test_jan2024.pdf',
          fileSize: 1024768,
          isFavorite: false,
          tags: ['blood test', 'lab results'],
          processingStatus: 'completed'
        },
        snippet: `...Laboratory values within normal range. Patient reported ${query} prior to blood draw but this did not affect results...`,
        similarityScore: 0.72,
        matchedTerms: [query.toLowerCase(), 'reported', 'patient'],
        highlightedContent: `Patient reported <mark>${query}</mark> prior to blood draw`
      }
    ];

    return mockResults;
  },

  async uploadDocument(patientId: string, file: File, title?: string): Promise<DocumentUploadResponse> {
    // FastAPI endpoint: POST /api/documents/upload
    // In production: 
    // const formData = new FormData();
    // formData.append('file', file);
    // formData.append('patient_id', patientId);
    // if (title) formData.append('title', title);
    // await apiRequest('/api/documents/upload', { method: 'POST', body: formData }, true)
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const newDocument: Document = {
      id: Date.now().toString(),
      patientId,
      title: title || file.name.replace(/\.[^/.]+$/, ""),
      content: 'Document processing in progress. Content will be available once AI analysis is complete.',
      summary: 'Document uploaded successfully. AI processing will extract key information and generate summary.',
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

  async toggleFavorite(documentId: string): Promise<{ success: boolean; isFavorite: boolean }> {
    // FastAPI endpoint: PATCH /api/documents/{documentId}/favorite
    // In production: await apiRequest(`/api/documents/${documentId}/favorite`, { method: 'PATCH' }, true)
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return {
      success: true,
      isFavorite: Math.random() > 0.5 // Random for demo
    };
  },

  async deleteDocument(documentId: string): Promise<{ success: boolean; message: string }> {
    // FastAPI endpoint: DELETE /api/documents/{documentId}
    // In production: await apiRequest(`/api/documents/${documentId}`, { method: 'DELETE' }, true)
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      success: true,
      message: 'Document deleted successfully'
    };
  }
};

// Analytics API
export const analyticsApi = {
  async getDashboardStats(): Promise<{
    totalPatients: number;
    totalDocuments: number;
    monthlyConsultations: number;
    timeSaved: string;
    patientGrowth: number;
    documentGrowth: number;
  }> {
    // Mock dashboard statistics
    await new Promise(resolve => setTimeout(resolve, 400));
    
    return {
      totalPatients: 127,
      totalDocuments: 1429,
      monthlyConsultations: 89,
      timeSaved: '24h',
      patientGrowth: 12,
      documentGrowth: 8
    };
  }
};