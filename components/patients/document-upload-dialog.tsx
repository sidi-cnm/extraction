'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import { documentApi } from '@/lib/api';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  File,
  Image,
  FileType
} from 'lucide-react';

interface DocumentUploadDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientId: string;
  onSuccess: () => void;
}

export function DocumentUploadDialog({
  open,
  onOpenChange,
  patientId,
  onSuccess
}: DocumentUploadDialogProps) {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [documentTitle, setDocumentTitle] = useState('');

  const uploadMutation = useMutation({
    mutationFn: (data: { file: File; title?: string }) => 
      documentApi.uploadDocument(patientId, data.file, data.title),
    onSuccess: (data) => {
      toast.success('Document uploaded successfully!');
      onSuccess();
      resetForm();
    },
    onError: (error) => {
      toast.error('Upload failed. Please try again.');
      console.error('Upload error:', error);
      setUploadProgress(0);
    }
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    maxFiles: 1,
    onDrop: (acceptedFiles, rejectedFiles) => {
      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        if (rejection.errors.some(e => e.code === 'file-too-large')) {
          toast.error('File is too large. Maximum size is 10MB.');
        } else if (rejection.errors.some(e => e.code === 'file-invalid-type')) {
          toast.error('File type not supported. Please upload PDF, DOC, DOCX, TXT, or image files.');
        } else {
          toast.error('File upload failed. Please try again.');
        }
        return;
      }
      
      setUploadedFiles(acceptedFiles);
      if (acceptedFiles.length > 0 && !documentTitle) {
        // Auto-generate title from filename
        const fileName = acceptedFiles[0].name;
        const titleWithoutExtension = fileName.replace(/\.[^/.]+$/, "");
        setDocumentTitle(titleWithoutExtension);
      }
    }
  });

  const resetForm = () => {
    setUploadedFiles([]);
    setDocumentTitle('');
    setUploadProgress(0);
  };

  const handleUpload = async () => {
    if (uploadedFiles.length === 0) return;

    const file = uploadedFiles[0];
    
    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      await uploadMutation.mutateAsync({ 
        file, 
        title: documentTitle.trim() || undefined 
      });
      setUploadProgress(100);
      setTimeout(() => {
        onOpenChange(false);
        resetForm();
      }, 1500);
    } catch (error) {
      clearInterval(progressInterval);
      setUploadProgress(0);
    }
  };

  const removeFile = () => {
    setUploadedFiles([]);
    setDocumentTitle('');
  };

  const getFileIcon = (file: File) => {
    const type = file.type;
    if (type.startsWith('image/')) return <Image className="w-5 h-5 text-blue-500" />;
    if (type === 'application/pdf') return <FileType className="w-5 h-5 text-red-500" />;
    if (type.includes('word') || type.includes('document')) return <FileText className="w-5 h-5 text-blue-600" />;
    return <File className="w-5 h-5 text-gray-500" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>upload a document</DialogTitle>
          <DialogDescription>
            Upload medical documents for AI processing and analysis. Supported formats: PDF, DOC, DOCX, TXT, Images (max 10MB).
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Document Title */}
          <div className="space-y-2">
            <Label htmlFor="title">Document Title (Optional)</Label>
            <Input
              id="title"
              placeholder="Enter a descriptive title for this document"
              value={documentTitle}
              onChange={(e) => setDocumentTitle(e.target.value)}
              disabled={uploadMutation.isPending || uploadProgress > 0}
            />
            <p className="text-xs text-gray-500">
              If left empty, the filename will be used as the title
            </p>
          </div>

          {/* Upload Area */}
          {uploadedFiles.length === 0 && (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary bg-primary/5'
                  : 'border-gray-300 hover:border-primary hover:bg-gray-50'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              {isDragActive ? (
                <p className="text-primary font-medium">Drop the file here...</p>
              ) : (
                <div>
                  <p className="text-gray-600 mb-2">
                    <span className="font-medium">select your file</span> or drag and drop
                  </p>
                  <p className="text-sm text-gray-500 mb-4">
                    png, jpg, pdf, doc, docx, txt files supported
                  </p>
                  <Button variant="outline" type="button">
                    Browse Files
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Uploaded File */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-3">
              <Label>Selected File</Label>
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(file)}
                    <div>
                      <p className="font-medium text-sm">{file.name}</p>
                      <p className="text-xs text-gray-500">
                        {formatFileSize(file.size)} • {file.type.split('/')[1]?.toUpperCase() || 'Unknown'}
                      </p>
                    </div>
                  </div>
                  {uploadProgress === 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={removeFile}
                      className="text-gray-500 hover:text-red-600"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Upload Progress */}
          {uploadProgress > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">
                  {uploadProgress < 100 ? 'Uploading and processing...' : 'Upload complete!'}
                </span>
                <span className="text-gray-600">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="w-full" />
            </div>
          )}

          {/* Processing Info */}
          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-800">
                  <p className="font-medium">AI Processing in Progress</p>
                  <p>Our AI is analyzing the document content, extracting key medical information, and generating an intelligent summary for quick access.</p>
                </div>
              </div>
            </div>
          )}

          {uploadProgress === 100 && (
            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-green-800">
                  <p className="font-medium">Document processed successfully!</p>
                  <p>The document has been analyzed and is ready for search and review. You can now find it in the patient's document list.</p>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={uploadMutation.isPending || (uploadProgress > 0 && uploadProgress < 100)}
            >
              {uploadProgress === 100 ? 'Close' : 'Cancel'}
            </Button>
            <Button
              onClick={handleUpload}
              disabled={uploadedFiles.length === 0 || uploadMutation.isPending || uploadProgress > 0}
              className="medical-gradient text-white"
            >
              {uploadMutation.isPending ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </>
              )}
            </Button>
          </div>

          {/* File Requirements */}
          <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded-lg">
            <p className="font-medium mb-1">File Requirements:</p>
            <ul className="space-y-1">
              <li>• Maximum file size: 10MB</li>
              <li>• Supported formats: PDF, DOC, DOCX, TXT, PNG, JPG, JPEG</li>
              <li>• Documents will be processed with AI for content extraction</li>
              <li>• Processing typically takes 30-60 seconds</li>
            </ul>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}