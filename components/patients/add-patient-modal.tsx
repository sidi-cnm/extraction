// Modal component for adding new patients
// Includes form validation and API integration

'use client';

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { patientApi } from '@/lib/api-client';
import { CreatePatientData } from '@/types';
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
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { toast } from 'sonner';
import { User, Mail, Phone, Hash, Calendar, MapPin, UserPlus } from 'lucide-react';

interface AddPatientModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddPatientModal({ open, onOpenChange }: AddPatientModalProps) {
  const [formData, setFormData] = useState<CreatePatientData>({
    name: '',
    email: '',
    phone: '',
    enrollNumber: '',
    dateOfBirth: '',
    address: '',
    emergencyContact: ''
  });
  const [errors, setErrors] = useState<Partial<CreatePatientData>>({});
  
  const queryClient = useQueryClient();

  // Mutation for creating a new patient
  const createPatientMutation = useMutation({
    mutationFn: (data: CreatePatientData) => patientApi.createPatient({
      ...data,
      lastVisit: new Date().toISOString().split('T')[0] // Set today as last visit
    }),
    onSuccess: () => {
      toast.success('Patient added successfully!');
      // Invalidate and refetch patients list
      queryClient.invalidateQueries({ queryKey: ['patients'] });
      resetForm();
      onOpenChange(false);
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to add patient');
    }
  });

  // Form validation
  const validateForm = (): boolean => {
    const newErrors: Partial<CreatePatientData> = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }
    
    if (!formData.phone.trim()) {
      newErrors.phone = 'Phone number is required';
    }
    
    if (!formData.enrollNumber.trim()) {
      newErrors.enrollNumber = 'Enroll number is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    createPatientMutation.mutate(formData);
  };

  const handleInputChange = (field: keyof CreatePatientData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      email: '',
      phone: '',
      enrollNumber: '',
      dateOfBirth: '',
      address: '',
      emergencyContact: ''
    });
    setErrors({});
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <UserPlus className="h-5 w-5 text-primary" />
            <span>Add New Patient</span>
          </DialogTitle>
          <DialogDescription>
            Enter the patient's information to create a new record in the system.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name Field */}
          <div className="space-y-2">
            <Label htmlFor="name">Full Name *</Label>
            <div className="relative">
              <User className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="name"
                type="text"
                placeholder="Enter patient's full name"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                className={`pl-10 ${errors.name ? 'border-red-500' : ''}`}
                disabled={createPatientMutation.isPending}
              />
            </div>
            {errors.name && (
              <p className="text-sm text-red-600">{errors.name}</p>
            )}
          </div>

          {/* Email Field */}
          <div className="space-y-2">
            <Label htmlFor="email">Email Address *</Label>
            <div className="relative">
              <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="email"
                type="email"
                placeholder="patient@example.com"
                value={formData.email}
                onChange={(e) => handleInputChange('email', e.target.value)}
                className={`pl-10 ${errors.email ? 'border-red-500' : ''}`}
                disabled={createPatientMutation.isPending}
              />
            </div>
            {errors.email && (
              <p className="text-sm text-red-600">{errors.email}</p>
            )}
          </div>

          {/* Phone Field */}
          <div className="space-y-2">
            <Label htmlFor="phone">Phone Number *</Label>
            <div className="relative">
              <Phone className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="phone"
                type="tel"
                placeholder="+1 (555) 123-4567"
                value={formData.phone}
                onChange={(e) => handleInputChange('phone', e.target.value)}
                className={`pl-10 ${errors.phone ? 'border-red-500' : ''}`}
                disabled={createPatientMutation.isPending}
              />
            </div>
            {errors.phone && (
              <p className="text-sm text-red-600">{errors.phone}</p>
            )}
          </div>

          {/* Enroll Number Field */}
          <div className="space-y-2">
            <Label htmlFor="enrollNumber">Enroll Number *</Label>
            <div className="relative">
              <Hash className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="enrollNumber"
                type="text"
                placeholder="EN001"
                value={formData.enrollNumber}
                onChange={(e) => handleInputChange('enrollNumber', e.target.value)}
                className={`pl-10 ${errors.enrollNumber ? 'border-red-500' : ''}`}
                disabled={createPatientMutation.isPending}
              />
            </div>
            {errors.enrollNumber && (
              <p className="text-sm text-red-600">{errors.enrollNumber}</p>
            )}
          </div>

          {/* Date of Birth Field */}
          <div className="space-y-2">
            <Label htmlFor="dateOfBirth">Date of Birth</Label>
            <div className="relative">
              <Calendar className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="dateOfBirth"
                type="date"
                value={formData.dateOfBirth}
                onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
                className="pl-10"
                disabled={createPatientMutation.isPending}
              />
            </div>
          </div>

          {/* Address Field */}
          <div className="space-y-2">
            <Label htmlFor="address">Address</Label>
            <div className="relative">
              <MapPin className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="address"
                type="text"
                placeholder="123 Main Street, City, State"
                value={formData.address}
                onChange={(e) => handleInputChange('address', e.target.value)}
                className="pl-10"
                disabled={createPatientMutation.isPending}
              />
            </div>
          </div>

          {/* Emergency Contact Field */}
          <div className="space-y-2">
            <Label htmlFor="emergencyContact">Emergency Contact</Label>
            <div className="relative">
              <User className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                id="emergencyContact"
                type="text"
                placeholder="Contact Name - Phone Number"
                value={formData.emergencyContact}
                onChange={(e) => handleInputChange('emergencyContact', e.target.value)}
                className="pl-10"
                disabled={createPatientMutation.isPending}
              />
            </div>
          </div>

          {/* Error Display */}
          {createPatientMutation.error && (
            <ErrorMessage 
              message={createPatientMutation.error instanceof Error 
                ? createPatientMutation.error.message 
                : 'Failed to add patient'
              } 
            />
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={createPatientMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="medical-gradient text-white"
              disabled={createPatientMutation.isPending}
            >
              {createPatientMutation.isPending ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Adding Patient...
                </>
              ) : (
                <>
                  <UserPlus className="h-4 w-4 mr-2" />
                  Add Patient
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}